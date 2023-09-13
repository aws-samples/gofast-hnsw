# Go (fast) HNSW

Gofast-HNSW is a reference implementation of the [Hierarchical Navigable Small World graph](https://arxiv.org/pdf/1603.09320.pdf) paper by Malkov & Yashunin (2018) as a companion to the AWS presentation by Ben Duncan *"What you need to know about Vector Databases. From use-cases to a deep dive on the technology"*  

This project is designed for *educational & AWS instance benchmarking purposes only*. The project compares the speed of the HNSW graph for k-NN search, the impact of settings to create the HNSW graph (M, mmax, mmax0, efConstruction, efSearch) compared to a naive (brute-search) implementation.

> NOTE: For a production grade library and complete implementation of HNSW that takes advantage of CPU acceleration (SSE, AVX2, AVX-512) see [https://github.com/nmslib/hnswlib](https://github.com/nmslib/hnswlib)


# Running a Vector Database on AWS

For customers that wish to deploy a Vector database on AWS, [Amazon OpenSearch](https://aws.amazon.com/opensearch-service/) provides a k-Nearest Neighbor (k-NN) search feature built on HNSW, to help power recommendation use-cases (for example, an "other songs you might like" feature in a music application), through to image recognition, and fraud detection.

OpenSearch k-NN Search plugin:

* https://opensearch.org/docs/1.0/search-plugins/knn/approximate-knn/

For more details on the memory requirements for large scale vector use-cases:

* https://aws.amazon.com/blogs/big-data/choose-the-k-nn-algorithm-for-your-billion-scale-use-case-with-opensearch/


# Usage

Build (go > 1.21.0 recommended)

```
make build
```

Run the benchmark for 1 million vectors (16 dimensions)

```
./bin/vecbench -num 1000000 -m 16 -mmax 16 -mmax0 32 -ef 200 -size 16 -csvfile benchmarks/1m.csv
```

Example output:

```
Total searches 1000000
Total matches from ground Truth: 9654061
Average 10-NN precision: 0.965406
HNSW efSearch (70):

HNSW Stats:
h.M = 16
h.Mmax = 16
h.Mmax0 = 32
h.Efconstruction = 200
h.Ep = 447724
h.Maxlevel = 5
h.Heuristic = true
h.Ml = 0.360674

Number of nodes = 1000000
        Level 0, number of nodes 937441, number of connections 25710940, avg 25
        Level 1, number of nodes 58480, number of connections 1000928, avg 16
        Level 2, number of nodes 3828, number of connections 65248, avg 16
        Level 3, number of nodes 230, number of connections 4000, avg 16
        Level 4, number of nodes 18, number of connections 320, avg 16
        Level 5, number of nodes 2, number of connections 0, avg 0
Total number of node levels = 6
HNSW search complete in 122.544270 (secs)
HNSW search queries per second 8160.316251 (8 threaded)
HNSW search queries per second 1020.039531 (Single threaded)
```

Note, the benchmark tool will create the specified number of vectors in a HNSW graph, conduct a brute-search for every element to find the top k-NN (10) and used as a ground-truth reference.

Once complete a HNSW search will run for the entire dataset to find the k-NN with a stepped `efSearch` paramater (in 10 increments) to reach the HNSW `ef` paramater used to create the index. This is used to change the accuracy of the search and speed, demonstrating the queries per second (qps) that can be achieved.

## Benchmark

To benchmark the results open the Jupyter Notebook `benchmarks/gengraph.ipynb` and place the results of the benchmark for the specific instance-type in a CSV file, e.g `benchmarks/c7g.8xlarge.1m-m16-16d-200ef.csv` for comparison.

## Results

The following instance types are benchmarked

* c6a.8xlarge - `AMD EPYC 7R13 CPU`
* c7g.8xlarge - `Graviton3 CPU - ARM Neoverse-V1`
* c6g.8xlarge - `Graviton2 CPU - ARM Neoverse-N1`
* c6i.8xlarge - `Intel(R) Xeon(R) Platinum 8375C`
* c6i.4xlarge - `Intel(R) Xeon(R) Platinum 8375C`
* c5.4xlarge - `Intel(R) Xeon(R) Platinum 8275CL`
* c4.4xlarge - `Intel(R) Xeon(R) CPU E5-2666 v3`

### HNSW Search performance

Results plotting the performance on a HNSW search (1 million vectors with 16 dimensions)

![HNSW Search - All Instances](./benchmarks/hnsw_search_all_instances.png)

### Brute search performance

To illustrate the performance of HNSW over a naive (brute-search) implementation

![Brute Search - All Instances](./benchmarks/brute_search_all_instances.png)

### HNSW vs Brute Search speedup

To conclude the benchmarking comparison between a brute-search and HNSW using the `gofast-hnsw` reference implementation.

As an example using a `c6i.4xlarge` instance HNSW provides a 66x speed-up at 90% precision compared to a brute-search approach.

![HNSW Search - All Instances](./benchmarks/hnsw_vs_brute_search_speedup.png)
