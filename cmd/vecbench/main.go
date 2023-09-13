package main

import (
	"container/heap"
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
	"strings"
	"time"

	"github.com/aws-samples/gofast-hnsw/vectordb/hnsw"
	"github.com/aws-samples/gofast-hnsw/vectordb/queue"
	"github.com/aws-samples/gofast-hnsw/vectordb/vectors"

	. "github.com/klauspost/cpuid/v2"
)

type Stats struct {
	Dim  int
	Size int
	K    int

	M         int
	Mmax      int
	Mmax0     int
	Ef        int
	Heuristic bool
	EfSearch  int

	CpuType           string
	CpuPhysicalCores  int
	CpuThreadsPerCore int
	CpuLogicalCores   int
	CpuFeatures       string

	IndexBuildSecs   float64
	IndexBuildMulti  float64
	IndexBuildSingle float64

	BruteSearchSecs   float64
	BruteSearchMulti  float64
	BruteSearchSingle float64

	HNSWSearchSecs   float64
	HNSWSearchMulti  float64
	HNSWSearchSingle float64

	GroundTruthHits int
	HNSWPrecision   float64

	DateStart time.Time
	DateEnd   time.Time
}

func main() {

	profile := flag.String("profile", "", "Set to enabling profiling with specified filename (default.pgo)")

	csvfile := flag.String("csvfile", "", "Export results to CSV file (stats.csv)")
	save := flag.String("save", "", "Export index to disk (data/vector.gob")

	vecDim := flag.Int("size", 32, "Set vector dimensions")
	vecNum := flag.Int("num", 1024, "Set number of vectors")
	k := flag.Int("k", 10, "Number of results to return for k-NN")

	m := flag.Int("m", 8, "Max number of HNSW layers")
	mmax := flag.Int("mmax", 8, "Max number of graph connections")
	mmax0 := flag.Int("mmax0", 16, "Max number of graph connections at layer 0")
	ef := flag.Int("ef", 200, "Size of the dynamic candidate list during index creation")
	heuristic := flag.Bool("heuristic", true, "Enable HNSW heuristic for neighbour selection")

	groundtruth := flag.Bool("groundtruth", true, "Compare HNSW results with brute force (ground truth)")
	hnswsearch := flag.Bool("hnswsearch", true, "Search using HNSW algorithm")

	var newFile bool = false

	flag.Parse()

	csvWriter := &csv.Writer{}

	stats := Stats{}

	stats.Dim = *vecDim
	stats.Size = *vecNum
	stats.K = *k

	stats.M = *m
	stats.Mmax = *mmax
	stats.Mmax0 = *mmax0
	stats.Ef = *ef
	stats.Heuristic = *heuristic

	stats.DateStart = time.Now()

	stats.CpuType = CPU.BrandName
	stats.CpuPhysicalCores = CPU.PhysicalCores
	stats.CpuThreadsPerCore = CPU.ThreadsPerCore
	stats.CpuLogicalCores = CPU.LogicalCores
	stats.CpuFeatures = strings.Join(CPU.FeatureSet(), ":")

	if *csvfile != "" {

		var file *os.File

		// check if file exists
		_, err := os.Lstat(*csvfile)
		if err != nil {

			file, err = os.Create(*csvfile)
			newFile = true
			if err != nil {
				log.Fatal("Error creating file:", err)
			}

		} else {

			// Append CSV files
			file, err = os.OpenFile(*csvfile, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0o660)

		}

		defer file.Close()

		csvWriter = csv.NewWriter(file)
		defer csvWriter.Flush()

	}

	// Create a CSV writer

	if *profile != "" {

		//var profile = "default.pgo"
		// If debug mode enabled
		// Enable debugging for profiling
		f, perr := os.Create(*profile)

		if perr != nil {
			log.Fatal(perr)
		}

		f2, err := os.Create(fmt.Sprintf("%s.mem", profile))

		if err != nil {
			log.Fatal(err)
		}

		pprof.StartCPUProfile(f)
		defer func() {
			pprof.StopCPUProfile()
			pprof.WriteHeapProfile(f2)
		}()

	}

	vec, _ := vectors.GenerateRandomVectors(*vecNum, *vecDim)

	// Init our HNSW Graph
	h, err := hnsw.New(*m, *mmax, *mmax0, *ef, len(vec[0]))

	// Use heurisitc
	h.Heuristic = *heuristic

	if err != nil {
		log.Fatal(err)
	}

	start := time.Now()

	fmt.Printf("gofast-HNSW - benchmark tool %f.\n\n", hnsw.Version)

	fmt.Printf("Running benchmarks on CPU (%s)\n", CPU.BrandName)

	fmt.Printf("Creating HNSW index with %d vectors (%d dimensions)\n", *vecNum, *vecDim)

	resultChan, jobs, err := h.InsertConcurrent(len(vec))

	if err != nil {
		log.Fatal(err)
	}

	for i := 1; i <= len(vec)-1; i++ {
		//_, err := h.Insert(vec[i])

		if i%1000 == 0 {
			// Clear the current line
			fmt.Printf("\033[2K\r")
			fmt.Printf("Added %d records", i)
		}

		jobs <- vec[i]

		//fmt.Println(i)

		if err != nil {
			log.Fatal(err)
		}

	}

	fmt.Println()

	close(jobs)

	h.Wg.Wait()
	close(resultChan)

	end := time.Since(start)

	numQ := vecNum

	stats.IndexBuildSecs = end.Seconds()
	stats.IndexBuildMulti = float64(*numQ) / end.Seconds()
	stats.IndexBuildSingle = float64(*numQ) / end.Seconds() / float64(runtime.NumCPU())

	fmt.Printf("HNSW Graph built in %0.6f (secs)\n", stats.IndexBuildSecs)
	fmt.Printf("HNSW Graph inserts per second %0.6f (%d threaded)\n", stats.IndexBuildMulti, runtime.NumCPU())
	fmt.Printf("HNSW Graph inserts per second %0.6f (Single thread)\n\n", stats.IndexBuildSingle)

	fmt.Printf("HNSW enter-point (Ep) => %d\n", h.Ep)
	fmt.Printf("Maxlevel => %d\n\n", h.Maxlevel)

	groundResults := make([][]uint32, 0)
	var hitSuccess int = 0

	if *groundtruth == true {

		fmt.Printf("Building Ground truth (brute) search for (%d) records. Returning (%d-NN) hits\n", *numQ, *k)

		start = time.Now()

		bruteSearchChan, bruteSearchJobs, err := h.BruteSearchConcurrent(*numQ, *k, runtime.NumCPU())

		if err != nil {
			log.Fatal(err)
		}

		groundResults = make([][]uint32, *numQ+1)

		for i := 0; i < *numQ; i++ {
			bruteSearchJobs <- hnsw.SearchQuery{Id: i, Qp: vec[i]}

			if i%1000 == 0 {
				// Clear the current line
				fmt.Printf("\033[2K\r")
				fmt.Printf("Searched %d records", i)
			}

		}

		close(bruteSearchJobs)

		h.Wg.Wait()
		close(bruteSearchChan)

		for result := range bruteSearchChan {

			for i3 := *k - 1; i3 >= 0; i3-- {

				if result.BestCandidates.Len() == 0 {
					break
				}

				groundResults[result.Id] = make([]uint32, *k)

				for i2 := *k - 1; i2 >= 0; i2-- {
					if result.BestCandidates.Len() > 0 {
						item := heap.Pop(&result.BestCandidates).(*queue.Item)
						groundResults[result.Id][i2] = item.Node
					}
				}

			}

		}

		fmt.Println("\nBrute Search Stats:\n")

		end = time.Since(start)

		stats.BruteSearchSecs = end.Seconds()
		stats.BruteSearchMulti = float64(*numQ) / end.Seconds()
		stats.BruteSearchSingle = float64(*numQ) / end.Seconds() / float64(runtime.NumCPU())

		fmt.Printf("Brute search complete in %0.6f (secs)\n", stats.BruteSearchSecs)

		fmt.Printf("Brute search queries per second %0.6f (%d threaded)\n", stats.BruteSearchMulti, runtime.NumCPU())
		fmt.Printf("Brute search queries per second %0.6f (Single threaded)\n", stats.BruteSearchSingle)

		fmt.Println("================================")

	}

	if *hnswsearch == true {

		for efSearch := 10; efSearch <= h.Efconstruction; efSearch += 10 {
			hitSuccess = 0
			stats.EfSearch = efSearch

			fmt.Printf("HNSW efSearch (%d):\n", efSearch)
			start = time.Now()

			totalSearch := 0

			searchChan, searchJobs, err := h.SearchConcurrent(*numQ, *k, efSearch, runtime.NumCPU())

			if err != nil {
				log.Fatal(err)
			}

			for i := 0; i < *numQ; i++ {
				searchJobs <- hnsw.SearchQuery{Id: i, Qp: vec[i]}

				if i%1000 == 0 {
					// Clear the current line
					fmt.Printf("\033[2K\r")
					fmt.Printf("Searched %d records", i)
				}

			}

			close(searchJobs)

			h.Wg.Wait()
			close(searchChan)

			end = time.Since(start)

			for result := range searchChan {

				for i3 := *k - 1; i3 >= 0; i3-- {

					if result.BestCandidates.Len() == 0 {
						//fmt.Println("No matches")
						break
					}

					item := heap.Pop(&result.BestCandidates).(*queue.Item)
					totalSearch++

					if *groundtruth == true {

						for k := *k - 1; k >= 0; k-- {

							if item.Node == groundResults[result.Id][k] {
								hitSuccess++
							}

						}

					}
				}

			}

			fmt.Println("HNSW Stats:")

			h.Stats()

			stats.HNSWSearchSecs = end.Seconds()
			stats.HNSWSearchMulti = float64(*numQ) / end.Seconds()
			stats.HNSWSearchSingle = float64(*numQ) / end.Seconds() / float64(runtime.NumCPU())

			fmt.Printf("HNSW search complete in %0.6f (secs)\n", stats.HNSWSearchSecs)

			fmt.Printf("HNSW search queries per second %0.6f (%d threaded)\n", stats.HNSWSearchMulti, runtime.NumCPU())
			fmt.Printf("HNSW search queries per second %0.6f (Single threaded)\n", stats.HNSWSearchSingle)

			fmt.Println("================================")

			stats.GroundTruthHits = hitSuccess
			stats.HNSWPrecision = float64(hitSuccess) / (float64(*numQ) * float64(*k))

			fmt.Printf("Total searches %d\n", stats.Size)
			fmt.Printf("Total matches from ground Truth: %d\n", stats.GroundTruthHits)
			fmt.Printf("Average 10-NN precision: %0.6f\n", stats.HNSWPrecision)

			// Optional, save our results
			stats.DateEnd = time.Now()

			// Write CSV file
			if *csvfile != "" {

				// If we are a new file, add the header
				if newFile == true {
					newFile = false
					header := []string{
						"Dim",
						"Size",
						"K",
						"M",
						"Mmax",
						"Mmax0",
						"Ef",
						"EfSearch",
						"Heuristic",
						"CpuType",
						"CpuPhysicalCores",
						"CpuThreadsPerCore",
						"CpuLogicalCores",
						"CpuFeatures",
						"IndexBuildSecs",
						"IndexBuildMulti",
						"IndexBuildSingle",
						"BruteSearchSecs",
						"BruteSearchMulti",
						"BruteSearchSingle",
						"HNSWSearchSecs",
						"HNSWSearchMulti",
						"HNSWSearchSingle",
						"GroundTruthHits",
						"HNSWPrecision",
						"DateStart",
						"DateEnd",
					}

					err = csvWriter.Write(header)

					if err != nil {
						log.Fatal("Error writing header to CSV:", err)
					}
				}

				err := csvWriter.Write([]string{
					fmt.Sprintf("%d", stats.Dim),
					fmt.Sprintf("%d", stats.Size),
					fmt.Sprintf("%d", stats.K),
					fmt.Sprintf("%d", stats.M),
					fmt.Sprintf("%d", stats.Mmax),
					fmt.Sprintf("%d", stats.Mmax0),
					fmt.Sprintf("%d", stats.Ef),
					fmt.Sprintf("%d", stats.EfSearch),
					fmt.Sprintf("%v", stats.Heuristic),

					stats.CpuType,
					fmt.Sprintf("%d", stats.CpuPhysicalCores),
					fmt.Sprintf("%d", stats.CpuThreadsPerCore),
					fmt.Sprintf("%d", stats.CpuLogicalCores),

					stats.CpuFeatures,

					fmt.Sprintf("%0.6f", stats.IndexBuildSecs),
					fmt.Sprintf("%0.6f", stats.IndexBuildMulti),
					fmt.Sprintf("%0.6f", stats.IndexBuildSingle),

					fmt.Sprintf("%0.6f", stats.BruteSearchSecs),
					fmt.Sprintf("%0.6f", stats.BruteSearchMulti),
					fmt.Sprintf("%0.6f", stats.BruteSearchSingle),

					fmt.Sprintf("%0.6f", stats.HNSWSearchSecs),
					fmt.Sprintf("%0.6f", stats.HNSWSearchMulti),
					fmt.Sprintf("%0.6f", stats.HNSWSearchSingle),

					fmt.Sprintf("%d", stats.GroundTruthHits),
					fmt.Sprintf("%0.6f", stats.HNSWPrecision),

					fmt.Sprintf("%s", stats.DateStart),
					fmt.Sprintf("%s", stats.DateEnd),
				},
				)

				if err != nil {
					log.Fatal("Error writing record to CSV:", err)
				}

			}

		}
	}

	// Export index to disk if specified
	if *save != "" {
		h.Save(*save)
	}

}
