package queue_test

import (
	"container/heap"
	"testing"

	"github.com/aws-samples/gofast-hnsw/vectordb/queue"
	"github.com/stretchr/testify/assert"
)

// Some items and their priorities.
var items = []float32{0.4, 9, 0.001, 0.0534, 0.234, 2.03, 2.042, 2.532, 1.0009, 0.329, 0.193, 0.999, 0.020391, 2.0991, 1.203, 10.03, 1.039, 1.0008, 5.029, 0.789}

func Test_MaxValidation(t *testing.T) {

	// Create a priority queue, put the items in it, and
	// establish the priority queue (heap) invariants.
	h := &queue.PriorityQueue{}
	h.Order = true // Sort max-heap, set false min-heap
	heap.Init(h)

	//pq := make(PriorityQueue, len(items))

	for k, v := range items {
		v := &queue.Item{
			Node:     uint32(k),
			Distance: v,
		}

		heap.Push(h, v)

	}

	// Confirm the min element is as we expect
	maxItem := h.Top().(*queue.Item)

	assert.Equal(t, float32(10.030000), maxItem.Distance)

	assert.Equal(t, uint32(15), maxItem.Node)

	assert.Equal(t, float32(10.030000), items[maxItem.Node])

	// Check len
	assert.Equal(t, 20, h.Len())

	// Prune

	maxEntries := 10

	for h.Len() > maxEntries {
		heap.Pop(h)
	}

	// Check len
	assert.Equal(t, 10, h.Len())

	// Confirm expected next element
	maxItem = h.Top().(*queue.Item)

	assert.Equal(t, float32(1.000800), maxItem.Distance)

	assert.Equal(t, uint32(17), maxItem.Node)

	assert.Equal(t, float32(1.000800), items[maxItem.Node])

	for h.Len() > 1 {
		heap.Pop(h)
	}

	// Check len
	assert.Equal(t, 1, h.Len())

	// Last remaining (smallest) element
	maxItem = h.Top().(*queue.Item)

	assert.Equal(t, float32(0.001000), maxItem.Distance)

	assert.Equal(t, uint32(2), maxItem.Node)

	assert.Equal(t, float32(0.001000), items[maxItem.Node])

	// Remove the last element, should be empty
	for h.Len() > 0 {
		heap.Pop(h)
	}

	assert.Equal(t, 0, h.Len())

}

func Test_MinValidation(t *testing.T) {

	// Create a priority queue, put the items in it, and
	// establish the priority queue (heap) invariants.
	h := &queue.PriorityQueue{}
	h.Order = false // Sort max-heap, set false min-heap
	heap.Init(h)

	//pq := make(PriorityQueue, len(items))

	for k, v := range items {
		v := &queue.Item{
			Node:     uint32(k),
			Distance: v,
		}

		heap.Push(h, v)

	}

	// Confirm the min element is as we expect
	maxItem := h.Top().(*queue.Item)

	assert.Equal(t, float32(0.001), maxItem.Distance)

	assert.Equal(t, uint32(2), maxItem.Node)

	assert.Equal(t, float32(0.001), items[maxItem.Node])

	// Check len
	assert.Equal(t, 20, h.Len())

	// Prune

	maxEntries := 10

	for h.Len() > maxEntries {
		heap.Pop(h)
	}

	// Check len
	assert.Equal(t, 10, h.Len())

	// Confirm expected next element
	maxItem = h.Top().(*queue.Item)

	assert.Equal(t, float32(1.000900), maxItem.Distance)

	assert.Equal(t, uint32(8), maxItem.Node)

	assert.Equal(t, float32(1.000900), items[maxItem.Node])

	for h.Len() > 1 {
		heap.Pop(h)
	}

	// Check len
	assert.Equal(t, 1, h.Len())

	// Last remaining (smallest) element
	maxItem = h.Top().(*queue.Item)

	assert.Equal(t, float32(10.03), maxItem.Distance)

	assert.Equal(t, uint32(15), maxItem.Node)

	assert.Equal(t, float32(10.03), items[maxItem.Node])

	// Remove the last element, should be empty
	for h.Len() > 0 {
		heap.Pop(h)
	}

	assert.Equal(t, 0, h.Len())

}
