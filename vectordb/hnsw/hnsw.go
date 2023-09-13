package hnsw

import (
	"container/heap"
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"

	"log"

	"github.com/aws-samples/gofast-hnsw/vectordb/distance"
	"github.com/aws-samples/gofast-hnsw/vectordb/queue"
	"github.com/willf/bitset"
)

type Node struct {
	Connections [][]uint32 // Links to other nodes
	Vectors     []float32  // Vector (X dimensions)
	Layer       int        // Layer the node exists in the HNSW tree
	Id          uint32     // Unique identifier
}

type NodeList struct {
	Nodes []Node
	mutex sync.RWMutex // Maintain a mutex for safe read/write access
}

type HNSW_Meta struct {
	Efconstruction int     // Size of the dynamic candidate list
	M              int     // Number of established connections (a reasonable range for M is from 5 to 48, smaller M generally produces better results for lower recalls and/or lower dimensional data. Bigger M is better for high recall and high dimensional data, and determines the memory consumption)
	Mmax           int     // Max number of connections per element/per layer (matches M)
	Mmax0          int     // Max for the 0 layer (Simulations suggest 2*M is a good choice, setting higher leads to performance degradation and excessive memory usage)
	Ml             float64 // Normalization factor for level generation
	Ep             int64   // Top layer of HNSW

	Maxlevel int // Track the current max level used

	Heuristic bool
}

type HNSW struct {
	Efconstruction int     // Size of the dynamic candidate list
	M              int     // Number of established connections (a reasonable range for M is from 5 to 48, smaller M generally produces better results for lower recalls and/or lower dimensional data. Bigger M is better for high recall and high dimensional data, and determines the memory consumption)
	Mmax           int     // Max number of connections per element/per layer (matches M)
	Mmax0          int     // Max for the 0 layer (Simulations suggest 2*M is a good choice, setting higher leads to performance degradation and excessive memory usage)
	Ml             float64 // Normalization factor for level generation
	Ep             int64   // Top layer of HNSW

	Maxlevel int // Track the current max level used

	Heuristic bool

	NodeList NodeList // Used to store the vectors within each node

	mutex sync.RWMutex
	Wg    sync.WaitGroup
}

type SearchQuery struct {
	Id int
	Qp []float32
}

type SearchResults struct {
	Id             int
	BestCandidates queue.PriorityQueue
}

// Set defaults for HNSW
// TODO: Auto select best values based on dataset and use ML model to determine
const M = 16
const Mmax = M
const Mmax0 = M * 2
const Efconstruction = 200 // Can be auto-configured using sample data

const Version = 1.0

func New(m int, mmax int, mmax0 int, efconstruction int, vecsize int) (h HNSW, err error) {

	h.M = m
	h.Mmax = mmax
	h.Mmax0 = mmax0
	h.Efconstruction = efconstruction

	// Init for special treatment of the first node
	h.Ep = 0
	h.Maxlevel = 0

	// Set to true to use heuristic algorithm (feature of HNSW), false to use naive K-NN (better for smaller datasets)
	h.Heuristic = true

	// Optimal Ml is 1/ln(M) which corresponds to an advantage of the controllable hierarchy for the overlap
	// on different layers to keep it small to reduce the average number of hops in a greedy search on each layer.
	h.Ml = 1 / math.Log(1.0*float64(h.M))

	// Populate our first node
	h.NodeList.Nodes = make([]Node, 1)

	// Required to create the first node and entry-point (TODO revise to use first element of the import, vs using null vector)
	h.NodeList.Nodes[0] = Node{Id: 0, Layer: 0, Vectors: make([]float32, vecsize), Connections: make([][]uint32, h.Mmax0+1)}

	return h, nil

}

// Input: Multi-layer graph hnsw, new element `q“, number of established connections `h.M“, max number of connections for each element per layer `h.Mmax“, size of the dynamic candidate list `h.efConstruction`, normalised factor for level generation `h.Ml`
// Output: update h inserting element q
func (h *HNSW) Insert(q []float32) (uint32, error) {

	var err error
	node := Node{}

	// TODO: Confirm performance difference
	node.Vectors = make([]float32, len(q))
	node.Vectors = q

	// Current distance from our starting-point (ep)
	currentObj := &h.NodeList.Nodes[h.Ep]
	currentDist, err := distance.L2_Opt(&currentObj.Vectors, &q)

	h.NodeList.mutex.Lock()

	// Generate the new layer
	node.Layer = int(math.Floor(-math.Log(rand.Float64()) * h.Ml))
	node.Id = uint32(len(h.NodeList.Nodes))

	// Create connections
	node.Connections = make([][]uint32, h.M+1)

	// Append new node
	h.NodeList.Nodes = append(h.NodeList.Nodes, node)

	h.NodeList.mutex.Unlock()

	ep := &queue.PriorityQueue{}
	ep.Order = false
	heap.Init(ep)

	var topCandidates queue.PriorityQueue
	topCandidates.Order = false

	// Find single shortest path from top layers above our current node, which will be our new starting-point

	for level := currentObj.Layer; level > node.Layer; level-- {

		changed := true

		for changed {
			changed = false

			// TODO: Must return the connections from our Ep to this specific level, otherwise will traverse the entire level which is inefficient
			for _, nodeId := range h.GetConnections(currentObj, level) {

				nodeDist, err := distance.L2_Opt(&h.NodeList.Nodes[nodeId].Vectors, &q)

				if err != nil {
					log.Fatal(err)
				}

				if nodeDist < currentDist {

					// Update the starting point to our new node
					currentObj = &h.NodeList.Nodes[nodeId]

					// Update the currently shortest distance
					currentDist = nodeDist

					changed = true

				}

			}

		}

	}

	// Update our min-heap for the winning node
	heap.Push(ep, &queue.Item{Distance: currentDist, Node: currentObj.Id})

	// For all levels equal and below our current node, find the top (closest) candidates and create a link
	for level := min(int(node.Layer), int(h.Maxlevel)); level >= 0; level-- {

		err = h.SearchLayer(&q, &queue.Item{Distance: currentDist, Node: currentObj.Id}, &topCandidates, int(h.Efconstruction), uint(level))

		if err != nil {
			log.Fatal(err)
		}

		// Switch type, naive k-NN, or Heuristic HNSW for linking nearest neighbours
		switch h.Heuristic {

		case false:

			// TODO: Confirm h.Max or h.Max0?
			h.SelectNeighboursSimple(&topCandidates, int(h.M))

		case true:
			// Select by heurisitc, using max-heap
			h.SelectNeighboursHeuristic(&topCandidates, int(h.M), false)

		}

		node.Connections[level] = make([]uint32, topCandidates.Len())

		for i := topCandidates.Len() - 1; i >= 0; i-- {
			candidate := heap.Pop(&topCandidates).(*queue.Item)
			//fmt.Printf("Adding node.Connections[%d][%d] = %d\n", level, i, candidate.Node)
			node.Connections[level][i] = candidate.Node
		}

	}

	// Append our new connections
	h.NodeList.mutex.Lock()
	h.NodeList.Nodes[node.Id].Connections = node.Connections
	h.NodeList.mutex.Unlock()

	// Next link the neighbour nodes to our new node, making it visible
	for level := min(int(node.Layer), int(h.Maxlevel)); level >= 0; level-- {

		h.NodeList.mutex.Lock()
		for _, neighbourNode := range h.NodeList.Nodes[node.Id].Connections[level] {
			h.AddConnections(neighbourNode, node.Id, level)
		}
		h.NodeList.mutex.Unlock()

	}

	if node.Layer > h.Maxlevel {
		// TODO: Revise, single mutex at h.mutex vs h.NodeList?
		h.mutex.Lock()
		//fmt.Printf("Updating MaxLevel (%d) to %d\n", h.Maxlevel, node.Layer)
		h.Ep = int64(node.Id)
		h.Maxlevel = node.Layer
		h.mutex.Unlock()
	}

	return node.Id, nil
}

// Insert concurrent
func (h *HNSW) InsertConcurrent(size int) (resultChan chan uint32, jobs chan []float32, err error) {

	resultChan = make(chan uint32, size)

	//var wg sync.WaitGroup

	numWorkers := runtime.NumCPU()
	jobs = make(chan []float32, numWorkers)

	// Launch the workers
	for i := 1; i <= numWorkers; i++ {
		h.Wg.Add(1)
		go h.InsertWorker(i, jobs, resultChan)
	}

	return

}

func (h *HNSW) InsertWorker(id int, jobs <-chan []float32, resultChan chan<- uint32) error {

	for q := range jobs {
		id, err := h.Insert(q)

		if err != nil {
			return err
		}

		resultChan <- id

	}

	defer h.Wg.Done()

	return nil
}

// Add links between nodes in the HNSW graph
func (h *HNSW) AddConnections(neighbourNode uint32, newNode uint32, level int) {

	//fmt.Printf("AddConnections, neighbourNode (%d) => newNode (%d), level %d\n", neighbourNode, newNode, level)

	// Change `M` depending on our level
	var maxConnections int

	// HNSW allows double the connections for the bottom level (0)
	if level == 0 {
		maxConnections = int(h.Mmax0)
	} else {
		maxConnections = int(h.Mmax)
	}

	// Add a min-heap
	h.NodeList.Nodes[neighbourNode].Connections[level] = append(h.NodeList.Nodes[neighbourNode].Connections[level], newNode)

	currentConnections := len(h.NodeList.Nodes[neighbourNode].Connections[level])

	if currentConnections > maxConnections {

		switch h.Heuristic {

		case false:
			// Add the new candidate to our queue
			topCandidates := &queue.PriorityQueue{}
			topCandidates.Order = true // min-heap, set to true for max-heap
			heap.Init(topCandidates)

			// Loop through each current connection and add the the max-heap
			for i := 0; i < currentConnections; i++ {
				connectedNode := h.NodeList.Nodes[neighbourNode].Connections[level][i]
				distanceBetweenNodes, err := distance.L2_Opt(&h.NodeList.Nodes[neighbourNode].Vectors, &h.NodeList.Nodes[connectedNode].Vectors)

				if err != nil {
					log.Fatal(err)
				}

				heap.Push(topCandidates, &queue.Item{Node: connectedNode, Distance: distanceBetweenNodes})
			}

			// Next, prune the weaker links, we want the best performing
			h.SelectNeighboursSimple(topCandidates, maxConnections)

			// Next, reorder our connected nodes with the improved lower distances within the graph

			h.NodeList.Nodes[neighbourNode].Connections[level] = make([]uint32, maxConnections)

			// Order by best performing match (index 0) .. lowest
			for i := maxConnections - 1; i >= 0; i-- {
				node := heap.Pop(topCandidates).(*queue.Item)
				h.NodeList.Nodes[neighbourNode].Connections[level][i] = node.Node
			}

		case true:

			// Add the new candidate to our queue
			topCandidates := &queue.PriorityQueue{}
			topCandidates.Order = false // min-heap, set to true for max-heap
			heap.Init(topCandidates)

			// Loop through each current connection and add the the max-heap
			for i := 0; i < currentConnections; i++ {
				connectedNode := h.NodeList.Nodes[neighbourNode].Connections[level][i]
				distanceBetweenNodes, err := distance.L2_Opt(&h.NodeList.Nodes[neighbourNode].Vectors, &h.NodeList.Nodes[connectedNode].Vectors)

				if err != nil {
					log.Fatal(err)
				}

				item := &queue.Item{Node: connectedNode, Distance: distanceBetweenNodes}
				heap.Push(topCandidates, item)

				//fmt.Printf("\tCurrent connections, node %d, distance %f\n", item.Node, item.Distance)
			}

			// Next, prune the weaker links, we want the best performing
			h.SelectNeighboursSimple(topCandidates, maxConnections)
			//h.SelectNeighboursHeuristic(topCandidates, maxConnections, true)

			// Next, reorder our connected nodes with the improved lower distances within the graph
			h.NodeList.Nodes[neighbourNode].Connections[level] = make([]uint32, maxConnections)

			// Order by best performing match (index 0) .. lowest
			for i := 0; i < maxConnections; i++ {
				node := heap.Pop(topCandidates).(*queue.Item)
				h.NodeList.Nodes[neighbourNode].Connections[level][i] = node.Node
			}

		}

	}

}

// Get links for a desired entry-point (ep) at a specified layer in the HNSW graph.
func (h *HNSW) GetConnections(ep *Node, level int) []uint32 {

	// Return pointer for performance?
	return ep.Connections[level]

}

// Input: Query element `q`, enter point `ep`, `M` number of nearest to `q“ elements to return, layer number `layerNum`
// Output: `nearestElements` closest neighbours to `q`
func (h *HNSW) SearchLayer(q *[]float32, ep *queue.Item, topCandidates *queue.PriorityQueue, ef int, level uint) (err error) {

	// TODO: Optimise
	//visited := make(map[uint32]bool)
	var visited bitset.BitSet

	// Add the new candidate to our queue
	candidates := &queue.PriorityQueue{}
	candidates.Order = false // min-heap, set to true for max-heap
	heap.Init(candidates)
	heap.Push(candidates, ep)

	// Init our topCandidates min-heap, first record worst distance
	topCandidates.Order = true // max-heap
	heap.Init(topCandidates)
	heap.Push(topCandidates, ep)

	for candidates.Len() > 0 {

		lowerBound := topCandidates.Top().(*queue.Item).Distance

		candidate := heap.Pop(candidates).(*queue.Item)

		if candidate.Distance > lowerBound {
			break
		}

		// Loop through each element in our nodes connections
		// TODO: Optimise loop, only add levels to connections if used, vs allocting all

		//h.NodeList.mutex.RLock()
		for _, node := range h.NodeList.Nodes[candidate.Node].Connections[level] {

			// If the node is not yet visited
			//if !visited[node] {
			if !visited.Test(uint(node)) {
				visited.Set(uint(node))
				//visited[node] = true

				nodeDist, err := distance.L2_Opt(q, &h.NodeList.Nodes[node].Vectors)

				// Required?
				if err != nil {
					log.Fatal(err)
				}

				item := &queue.Item{
					Distance: nodeDist,
					Node:     node,
				}

				topDistance := topCandidates.Top().(*queue.Item).Distance

				// Add the element to topCandidates if size < efConstruction
				if topCandidates.Len() < ef {

					if node != ep.Node {
						heap.Push(topCandidates, item)
					}

					// Add our new node to our list of candidates to search
					heap.Push(candidates, item)

				} else if topDistance > nodeDist {

					heap.Push(topCandidates, item)

					// Remove the worst performing
					heap.Pop(topCandidates)

					// Add our new node to our list of candidates to search
					heap.Push(candidates, item)

				}

			}

		}
		//h.NodeList.mutex.RUnlock()

	}

	return nil

}

// Input: Candidate elements `C`, number of neighbours to return `M`
// Output: `M` nearest elements in heap
func (h *HNSW) SelectNeighboursSimple(topCandidates *queue.PriorityQueue, M int) {

	for topCandidates.Len() > M {
		_ = heap.Pop(topCandidates).(*queue.Item)
	}

}

// Input: base element q, candidate elements C, number of neighbors to return M, layer number lc, flag indicating whether or not to extend candidate list extendCandidates, flag indicating whether or not to add discarded elements keepPrunedConnections
// Output: M elements selected by the heuristic

func (h *HNSW) SelectNeighboursHeuristic(topCandidates *queue.PriorityQueue, M int, order bool) {

	// If results < M, return, nothing required
	if topCandidates.Len() < M {
		return
	}

	// Create our new priority queues
	newCandidates := &queue.PriorityQueue{}

	tmpCandidates := queue.PriorityQueue{}
	tmpCandidates.Order = order // min-heap, set to true for max-heap
	heap.Init(&tmpCandidates)

	items := make([]*queue.Item, 0, M)

	if !order {

		newCandidates.Order = order // min-heap, set to true for max-heap
		heap.Init(newCandidates)

		// Add existing candidates to our new queue
		for topCandidates.Len() > 0 {

			item := heap.Pop(topCandidates).(*queue.Item)

			//fmt.Printf("Pushing newCandidates, node (%d), distance (%f)\n", item.Node, item.Distance)
			heap.Push(newCandidates, item)
			//fmt.Printf("\t item %d, distance (%f)\n", item.Node, item.Distance)
		}

	} else {
		newCandidates = topCandidates
	}

	//fmt.Println()

	// Scan through our new queue (order changed from min-heap > max-heap or vice-versa depending on order arg)
	for newCandidates.Len() > 0 {

		// Finish if items reaches our desired length
		if len(items) >= M {
			//fmt.Printf("Breaking %d > %d\n", len(items), M)
			break
		}

		item := heap.Pop(newCandidates).(*queue.Item)

		hit := true

		// Search through each item and determine if distance from node lower for items in set
		for _, v := range items {

			nodeDist, _ := distance.L2_Opt(&h.NodeList.Nodes[v.Node].Vectors, &h.NodeList.Nodes[item.Node].Vectors)

			if nodeDist < item.Distance {
				//fmt.Printf("SelectNeighboursHeuristic, node %d to candidate %d distance (%f), lower than original %f\n", v.Node, item.Node, nodeDist, item.Distance)
				hit = false
				break
			}

		}

		if hit {
			items = append(items, item)
		} else {
			heap.Push(&tmpCandidates, item)
		}

	}

	// Add any additional items from tmpCandidates if current items < M
	for len(items) < M && tmpCandidates.Len() > 0 {
		//fmt.Printf("SelectNeighboursHeuristic, items len (%d), tmpCandidates len %d, appending item from tmpCandidates\n", len(items), tmpCandidates.Len())

		item := heap.Pop(&tmpCandidates).(*queue.Item)
		items = append(items, item)
	}

	// Last step, append our results into our original min/max-heap
	for _, item := range items {
		heap.Push(topCandidates, item)
	}

}

func (h *HNSW) LegacySearchLayer(q *[]float32, ep *[]float32, C *[]uint32, M int) queue.PriorityQueue {

	var lowerBound float32

	// Find the lowerBound based on our entry-point
	lowerBound, err := distance.L2_Opt(q, ep)

	if err != nil {
		log.Fatal(err)
	}

	topCandidates := &queue.PriorityQueue{}

	topCandidates.Order = false // min-heap, set to true for max-heap
	heap.Init(topCandidates)

	var id int

	for _, node := range *C {

		nodeDist, err := distance.L2_Opt(q, &h.NodeList.Nodes[node].Vectors)

		if err != nil {
			log.Fatal(err)
		}

		if topCandidates.Len() < M || lowerBound > nodeDist {

			candidate := &queue.Item{
				Distance: nodeDist,
				Node:     node,
			}

			// Add the new candidate to our queue
			heap.Push(topCandidates, candidate)

			// If number of candidates > efConstruction, pop the smallest distance out
			if topCandidates.Len() > int(h.Efconstruction) {
				heap.Pop(topCandidates)
			}

			// Update the lowerBound to our min value in the heap
			if topCandidates.Len() > 0 {
				lowerBound = topCandidates.Items[0].Distance
			}

			id++
		}

	}

	return *topCandidates

}

// Search layer
// Input: Query element `q`, number of nearest neighbours to return `K`, size of the dynamic candidate list `ef`
// Output: `nearestElements` to `q`
func (h *HNSW) KnnSearch(q Node, K int, ef int) (nearestElements []Node) {

	return

}

// Find query point `q` and result `K` results (max-heap)
func (h *HNSW) Search(q *[]float32, topCandidates *queue.PriorityQueue, K int, efSearch int) (err error) {

	currentObj := &h.NodeList.Nodes[h.Ep]
	match, currentDist, err := h.FindEp(q, currentObj, 0)

	if err != nil {
		log.Fatal(err)
	}

	err = h.SearchLayer(q, &queue.Item{Distance: currentDist, Node: match.Id}, topCandidates, efSearch, 0)

	if err != nil {
		log.Fatal(err)
	}

	for topCandidates.Len() > K {
		_ = heap.Pop(topCandidates).(*queue.Item)
	}

	return nil
}

// Search concurrent
func (h *HNSW) SearchConcurrent(size int, K int, efSearch int, numWorkers int) (resultChan chan SearchResults, jobs chan SearchQuery, err error) {

	resultChan = make(chan SearchResults, size)

	//var wg sync.WaitGroup

	jobs = make(chan SearchQuery, numWorkers)

	// Launch the workers
	for i := 1; i <= numWorkers; i++ {
		h.Wg.Add(1)
		go h.SearchWorker(i, K, efSearch, jobs, resultChan)
	}

	return

}

func (h *HNSW) SearchWorker(id int, K int, efSearch int, jobs <-chan SearchQuery, resultChan chan<- SearchResults) error {

	for q := range jobs {

		var bestCandidates queue.PriorityQueue
		heap.Init(&bestCandidates)
		err := h.Search(&q.Qp, &bestCandidates, K, efSearch)

		if err != nil {
			return err
		}

		resultChan <- SearchResults{Id: q.Id, BestCandidates: bestCandidates}

	}

	defer h.Wg.Done()

	return nil
}

// Brute search
func (h *HNSW) BruteSearch(q *[]float32, K int) (topCandidates queue.PriorityQueue, err error) {

	topCandidates.Order = true

	h.NodeList.mutex.RLock()

	for i := 0; i < len(h.NodeList.Nodes); i++ {

		nodeDist, err := distance.L2_Opt(q, &h.NodeList.Nodes[i].Vectors)

		if err != nil {
			log.Fatal(err)
		}

		if topCandidates.Len() < K {
			heap.Push(&topCandidates, &queue.Item{
				Node:     uint32(i),
				Distance: nodeDist,
			})
			continue
		}

		largestDist := topCandidates.Top().(*queue.Item)

		if nodeDist < largestDist.Distance {
			heap.Pop(&topCandidates)
			heap.Push(&topCandidates, &queue.Item{
				Node:     uint32(i),
				Distance: nodeDist,
			})
		}

	}

	h.NodeList.mutex.RUnlock()

	return
}

// Search concurrent
func (h *HNSW) BruteSearchConcurrent(size int, K int, numWorkers int) (resultChan chan SearchResults, jobs chan SearchQuery, err error) {

	resultChan = make(chan SearchResults, size)

	//var wg sync.WaitGroup

	jobs = make(chan SearchQuery, numWorkers)

	// Launch the workers
	for i := 1; i <= numWorkers; i++ {
		h.Wg.Add(1)
		go h.BruteSearchWorker(i, K, jobs, resultChan)
	}

	return

}

func (h *HNSW) BruteSearchWorker(id int, K int, jobs <-chan SearchQuery, resultChan chan<- SearchResults) error {

	for q := range jobs {

		//var bestCandidates queue.PriorityQueue
		//heap.Init(&bestCandidates)
		bestCandidates, err := h.BruteSearch(&q.Qp, K)

		if err != nil {
			return err
		}

		resultChan <- SearchResults{Id: q.Id, BestCandidates: bestCandidates}

	}

	defer h.Wg.Done()

	return nil
}

func (h *HNSW) FindEp(q *[]float32, currentObj *Node, layer int16) (match Node, currentDist float32, err error) {

	currentDist, err = distance.L2_Opt(q, &currentObj.Vectors)

	// Find single shortest path from top layers above our current node, which will be our new starting-point
	for level := h.Maxlevel; level > 0; level-- {

		scan := true

		for scan {
			scan = false

			// TODO: Must return the connections from our Ep to this specific level, otherwise will traverse the entire level which is inefficient

			for _, nodeId := range h.GetConnections(currentObj, level) {

				nodeDist, err := distance.L2_Opt(&h.NodeList.Nodes[nodeId].Vectors, q)

				if err != nil {
					log.Fatal(err)
				}

				if nodeDist < currentDist {

					// Update the starting point to our new node
					match = h.NodeList.Nodes[nodeId]

					// Update the currently shortest distance
					currentDist = nodeDist

					// If a smaller match found, continue
					scan = true

				}

			}

		}

	}

	return match, currentDist, nil

}

func (h *HNSW) Stats() {

	fmt.Printf("h.M = %d\n", h.M)
	fmt.Printf("h.Mmax = %d\n", h.Mmax)
	fmt.Printf("h.Mmax0 = %d\n", h.Mmax0)
	fmt.Printf("h.Efconstruction = %d\n", h.Efconstruction)

	fmt.Printf("h.Ep = %d\n", h.Ep)
	fmt.Printf("h.Maxlevel = %d\n", h.Maxlevel)

	fmt.Printf("h.Heuristic = %v\n", h.Heuristic)

	fmt.Printf("h.Ml = %f\n\n", h.Ml)

	fmt.Printf("Number of nodes = %d\n", len(h.NodeList.Nodes))

	levelStats := make([]int, h.Maxlevel+1)
	connectionStats := make([]int, h.Maxlevel+1)
	connectionNodeStats := make([]int, h.Maxlevel+1)

	for i := 0; i < len(h.NodeList.Nodes)-1; i++ {
		levelStats[h.NodeList.Nodes[i].Layer]++

		// Loop through each connection
		for i2 := int(h.NodeList.Nodes[i].Layer); i2 >= 0; i2-- {

			if len(h.NodeList.Nodes[i].Connections[i2]) > i2 {
				total := len(h.NodeList.Nodes[i].Connections[i2])
				connectionStats[i2] += total
				connectionNodeStats[i2]++

			}

		}
	}

	for k, v := range levelStats {
		avg := int(connectionStats[k] / max(1, connectionNodeStats[k]))
		fmt.Printf("\tLevel %d, number of nodes %d, number of connections %d, avg %d\n", k, v, connectionStats[k], avg)
	}

	fmt.Printf("Total number of node levels = %d\n", len(levelStats))

}

// Peek a node
func (h *HNSW) PeekNode(id int) Node {

	return h.NodeList.Nodes[id]

}

func (h *HNSW) Save(filename string) (err error) {

	// Dump meta-data
	file, err := os.Create(fmt.Sprintf("%s.meta", filename))

	if err != nil {
		return err
	}

	encoder := gob.NewEncoder(file)

	// Faster performance?
	/*
		for k, _ := range h.NodeList.Nodes {
			err = encoder.Encode(h.NodeList.Nodes[k])
		}
	*/

	err = encoder.Encode(HNSW_Meta{
		Efconstruction: h.Efconstruction,
		M:              h.M,
		Mmax:           h.Mmax,
		Mmax0:          h.Mmax0,
		Ml:             h.Ml,
		Ep:             h.Ep,
		Maxlevel:       h.Maxlevel,
		Heuristic:      h.Heuristic,
	})

	if err != nil {
		return err
	}

	file.Close()

	file, err = os.Create(filename)

	if err != nil {
		return err
	}

	encoder = gob.NewEncoder(file)

	// Faster performance?
	/*
		for k, _ := range h.NodeList.Nodes {
			err = encoder.Encode(h.NodeList.Nodes[k])
		}
	*/

	err = encoder.Encode(h.NodeList.Nodes)

	if err != nil {
		return err
	}

	file.Close()

	return

}

func Load(filename string) (h HNSW, err error) {

	file, err := os.Open(fmt.Sprintf("%s.meta", filename))

	if err != nil {
		return
	}

	decoder := gob.NewDecoder(file)

	meta := HNSW_Meta{}

	err = decoder.Decode(&meta)

	h.Efconstruction = meta.Efconstruction
	h.M = meta.M
	h.Mmax = meta.Mmax
	h.Mmax0 = meta.Mmax0
	h.Ml = meta.Ml
	h.Ep = meta.Ep
	h.Maxlevel = meta.Maxlevel
	h.Heuristic = meta.Heuristic

	if err != nil {
		return
	}

	file.Close()

	file, err = os.Open(filename)

	if err != nil {
		return
	}

	decoder = gob.NewDecoder(file)

	err = decoder.Decode(&h.NodeList.Nodes)

	if err != nil {
		return
	}

	file.Close()

	return

}

// Private functions

// Find the min value - Use Go 1.21, inbuilt?
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
