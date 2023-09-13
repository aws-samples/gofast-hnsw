package hnsw_test

import (
	"container/heap"
	"fmt"
	"log"
	"testing"

	"github.com/aws-samples/gofast-hnsw/vectordb/hnsw"
	"github.com/aws-samples/gofast-hnsw/vectordb/queue"
	"github.com/aws-samples/gofast-hnsw/vectordb/vectors"
	"github.com/stretchr/testify/assert"
)

type TestCases struct {
	VectorSize int
	VectorDim  int

	M              int
	Efconstruction int
	Heuristic      bool
	K              int

	Precision  float64
	Concurrent bool
}

func Test_New(t *testing.T) {

	h, err := hnsw.New(8, 8, 16, 200, 1024)

	assert.Nil(t, err)

	assert.Equal(t, uint16(8), h.M)
	assert.Equal(t, uint16(8), h.Mmax)
	assert.Equal(t, uint16(16), h.Mmax0)
	assert.Equal(t, uint16(200), h.Efconstruction)

	assert.Equal(t, 1, len(h.NodeList.Nodes))

}

func Test_ValidateInsertSearch(t *testing.T) {

	tests := []TestCases{
		{
			VectorSize:     1000,
			VectorDim:      16,
			M:              8,
			Efconstruction: 200,
			Heuristic:      true,
			Precision:      0.99,
			K:              10,
			Concurrent:     false,
		},
		{
			VectorSize:     1000,
			VectorDim:      16,
			M:              8,
			Efconstruction: 200,
			Heuristic:      false,
			Precision:      0.99,
			K:              10,
			Concurrent:     false,
		},
		{
			VectorSize:     1000,
			VectorDim:      1024,
			M:              12,
			Efconstruction: 200,
			Heuristic:      true,
			Precision:      0.98,
			K:              10,
			Concurrent:     false,
		},

		{
			VectorSize:     10000,
			VectorDim:      16,
			M:              16,
			Efconstruction: 200,
			Heuristic:      true,
			Precision:      0.99,
			K:              10,
			Concurrent:     false,
		},
		{
			VectorSize:     10000,
			VectorDim:      32,
			M:              16,
			Efconstruction: 200,
			Heuristic:      true,
			Precision:      0.99,
			K:              10,
			Concurrent:     false,
		},

		/*
			{
				VectorSize:     100000,
				VectorDim:      16,
				M:              16,
				Efconstruction: 200,
				Heuristic:      true,
				Precision:      0.95,
				K:              10,
			},

				{
					VectorSize:     1000000,
					VectorDim:      32,
					M:              24,
					Efconstruction: 300,
					Heuristic:      true,
					Precision:      0.95,
					K:              10,
				},
		*/

		{
			VectorSize:     1000,
			VectorDim:      16,
			M:              8,
			Efconstruction: 200,
			Heuristic:      true,
			Precision:      0.99,
			K:              10,
			Concurrent:     true,
		},
		{
			VectorSize:     1000,
			VectorDim:      16,
			M:              8,
			Efconstruction: 200,
			Heuristic:      false,
			Precision:      0.99,
			K:              10,
			Concurrent:     true,
		},
		{
			VectorSize:     1000,
			VectorDim:      1024,
			M:              12,
			Efconstruction: 200,
			Heuristic:      true,
			Precision:      0.98,
			K:              10,
			Concurrent:     true,
		},

		{
			VectorSize:     10000,
			VectorDim:      16,
			M:              16,
			Efconstruction: 200,
			Heuristic:      true,
			Precision:      0.99,
			K:              10,
			Concurrent:     true,
		},
		{
			VectorSize:     10000,
			VectorDim:      32,
			M:              16,
			Efconstruction: 200,
			Heuristic:      true,
			Precision:      0.99,
			K:              10,
			Concurrent:     true,
		},
	}

	fmt.Println(tests)

	for _, tc := range tests {

		testname := fmt.Sprintf("Vec=%d,Dim=%d,Heuristic=%t,M=%d,Concurrent=%t,Precision=%f", tc.VectorSize, tc.VectorDim, tc.Heuristic, tc.M, tc.Concurrent, tc.Precision)

		t.Run(testname, func(t *testing.T) {

			vecs, err := vectors.GenerateRandomVectors(tc.VectorSize, tc.VectorDim)

			assert.Nil(t, err)

			assert.Equal(t, tc.VectorSize, len(vecs))
			assert.Equal(t, tc.VectorDim, len(vecs[0]))

			h, err := hnsw.New(tc.M, tc.M, tc.M*2, tc.Efconstruction, len(vecs[0]))

			assert.Nil(t, err)

			h.Heuristic = tc.Heuristic

			resultChan := make(chan uint32)
			jobs := make(chan []float32)

			if tc.Concurrent {
				resultChan, jobs, err = h.InsertConcurrent(len(vecs))
				assert.Nil(t, err)
			}

			for i := 0; i < len(vecs); i++ {

				if tc.Concurrent {
					jobs <- vecs[i]
				} else {
					id, err := h.Insert(vecs[i])

					assert.GreaterOrEqual(t, id, uint32(0))
					assert.Nil(t, err)
				}

			}

			if tc.Concurrent {
				close(jobs)

				h.Wg.Wait()
				close(resultChan)
			}

			groundResults := make([][]uint32, len(vecs))

			for i := 0; i < len(vecs); i++ {

				bestCandidatesBrute, _ := h.BruteSearch(&vecs[i], tc.K)

				groundResults[i] = make([]uint32, tc.K)

				for i2 := tc.K - 1; i2 >= 0; i2-- {
					if bestCandidatesBrute.Len() > 0 {
						item := heap.Pop(&bestCandidatesBrute).(*queue.Item)
						groundResults[i][i2] = item.Node
					}
				}

			}

			hitSuccess := 0
			totalSearch := 0

			for i := 0; i < len(vecs); i++ {
				var bestCandidates queue.PriorityQueue
				heap.Init(&bestCandidates)
				err = h.Search(&vecs[i], &bestCandidates, tc.K, tc.Efconstruction)

				if err != nil {
					log.Fatal(err)
				}

				for i3 := tc.K - 1; i3 >= 0; i3-- {

					if bestCandidates.Len() == 0 {
						fmt.Println("No matches")
						break
					}

					item := heap.Pop(&bestCandidates).(*queue.Item)
					totalSearch++

					for k := tc.K - 1; k >= 0; k-- {

						if item.Node == groundResults[i][k] {
							hitSuccess++
						}

					}

				}

			}

			precision := float64(hitSuccess) / (float64(len(vecs)) * float64(tc.K))

			//fmt.Printf("Precision => %f\n", precision)
			assert.GreaterOrEqual(t, precision, float64(tc.Precision))

		})

	}

}
