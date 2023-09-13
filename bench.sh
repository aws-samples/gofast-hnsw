#!/bin/sh

# Run various benchmarks from 10k-1M vectors, between 16,32,64,128 dimensions.
make build

./bin/vecbench -num 10000 -m 8 -mmax 8 -mmax0 16 -ef 100 -size 16 -csvfile benchmarks/10k-m8-16d.csv
./bin/vecbench -num 100000 -m 8 -mmax 8 -mmax0 16 -ef 100 -size 16 -csvfile benchmarks/100k-m8-16d.csv
./bin/vecbench -num 200000 -m 8 -mmax 8 -mmax0 16 -ef 100 -size 16 -csvfile benchmarks/200k-m8-16d.csv
./bin/vecbench -num 300000 -m 8 -mmax 8 -mmax0 16 -ef 100 -size 16 -csvfile benchmarks/300k-m8-16d.csv
./bin/vecbench -num 400000 -m 8 -mmax 8 -mmax0 16 -ef 100 -size 16 -csvfile benchmarks/400k-m8-16d.csv
./bin/vecbench -num 500000 -m 8 -mmax 8 -mmax0 16 -ef 100 -size 16 -csvfile benchmarks/500k-m8-16d.csv
./bin/vecbench -num 600000 -m 8 -mmax 8 -mmax0 16 -ef 100 -size 16 -csvfile benchmarks/600k-m8-16d.csv
./bin/vecbench -num 700000 -m 8 -mmax 8 -mmax0 16 -ef 100 -size 16 -csvfile benchmarks/700k-m8-16d.csv
./bin/vecbench -num 800000 -m 8 -mmax 8 -mmax0 16 -ef 100 -size 16 -csvfile benchmarks/800k-m8-16d.csv
./bin/vecbench -num 900000 -m 8 -mmax 8 -mmax0 16 -ef 100 -size 16 -csvfile benchmarks/900k-m8-16d.csv
./bin/vecbench -num 1000000 -m 8 -mmax 8 -mmax0 16 -ef 100 -size 16 -csvfile benchmarks/1m-m8-16d.csv

./bin/vecbench -num 10000 -m 12 -mmax 12 -mmax0 24 -ef 200 -size 32 -csvfile benchmarks/10k-m12-32d.csv
./bin/vecbench -num 100000 -m 12 -mmax 12 -mmax0 24 -ef 200 -size 32 -csvfile benchmarks/100k-m12-32d.csv
./bin/vecbench -num 200000 -m 12 -mmax 12 -mmax0 24 -ef 200 -size 32 -csvfile benchmarks/200k-m12-32d.csv
./bin/vecbench -num 300000 -m 12 -mmax 12 -mmax0 24 -ef 200 -size 32 -csvfile benchmarks/300k-m12-32d.csv
./bin/vecbench -num 400000 -m 12 -mmax 12 -mmax0 24 -ef 200 -size 32 -csvfile benchmarks/400k-m12-32d.csv
./bin/vecbench -num 500000 -m 12 -mmax 12 -mmax0 24 -ef 200 -size 32 -csvfile benchmarks/500k-m12-32d.csv
./bin/vecbench -num 600000 -m 12 -mmax 12 -mmax0 24 -ef 200 -size 32 -csvfile benchmarks/600k-m12-32d.csv
./bin/vecbench -num 700000 -m 12 -mmax 12 -mmax0 24 -ef 200 -size 32 -csvfile benchmarks/700k-m12-32d.csv
./bin/vecbench -num 800000 -m 12 -mmax 12 -mmax0 24 -ef 200 -size 32 -csvfile benchmarks/800k-m12-32d.csv
./bin/vecbench -num 900000 -m 12 -mmax 12 -mmax0 24 -ef 200 -size 32 -csvfile benchmarks/900k-m12-32d.csv
./bin/vecbench -num 1000000 -m 12 -mmax 12 -mmax0 24 -ef 200 -size 32 -csvfile benchmarks/1m-m12-32d.csv

./bin/vecbench -num 10000 -m 16 -mmax 16 -mmax0 32 -ef 300 -size 64 -csvfile benchmarks/10k-m16-64d.csv
./bin/vecbench -num 100000 -m 16 -mmax 16 -mmax0 32 -ef 300 -size 64 -csvfile benchmarks/100k-m16-64d.csv
./bin/vecbench -num 200000 -m 16 -mmax 16 -mmax0 32 -ef 300 -size 64 -csvfile benchmarks/200k-m16-64d.csv
./bin/vecbench -num 300000 -m 16 -mmax 16 -mmax0 32 -ef 300 -size 64 -csvfile benchmarks/300k-m16-64d.csv
./bin/vecbench -num 400000 -m 16 -mmax 16 -mmax0 32 -ef 300 -size 64 -csvfile benchmarks/400k-m16-64d.csv
./bin/vecbench -num 500000 -m 16 -mmax 16 -mmax0 32 -ef 300 -size 64 -csvfile benchmarks/500k-m16-64d.csv
./bin/vecbench -num 600000 -m 16 -mmax 16 -mmax0 32 -ef 300 -size 64 -csvfile benchmarks/600k-m16-64d.csv
./bin/vecbench -num 700000 -m 16 -mmax 16 -mmax0 32 -ef 300 -size 64 -csvfile benchmarks/700k-m16-64d.csv
./bin/vecbench -num 800000 -m 16 -mmax 16 -mmax0 32 -ef 300 -size 64 -csvfile benchmarks/800k-m16-64d.csv
./bin/vecbench -num 900000 -m 16 -mmax 16 -mmax0 32 -ef 300 -size 64 -csvfile benchmarks/900k-m16-64d.csv
./bin/vecbench -num 1000000 -m 16 -mmax 16 -mmax0 32 -ef 300 -size 64 -csvfile benchmarks/1m-m16-64d.csv

./bin/vecbench -num 10000 -m 16 -mmax 16 -mmax0 32 -ef 400 -size 128 -csvfile benchmarks/10k-m16-128d.csv
./bin/vecbench -num 100000 -m 16 -mmax 16 -mmax0 32 -ef 400 -size 128 -csvfile benchmarks/100k-m16-128d.csv
./bin/vecbench -num 200000 -m 16 -mmax 16 -mmax0 32 -ef 400 -size 128 -csvfile benchmarks/200k-m16-128d.csv
./bin/vecbench -num 300000 -m 16 -mmax 16 -mmax0 32 -ef 400 -size 128 -csvfile benchmarks/300k-m16-128d.csv
./bin/vecbench -num 400000 -m 16 -mmax 16 -mmax0 32 -ef 400 -size 128 -csvfile benchmarks/400k-m16-128d.csv
./bin/vecbench -num 500000 -m 16 -mmax 16 -mmax0 32 -ef 400 -size 128 -csvfile benchmarks/500k-m16-128d.csv
./bin/vecbench -num 600000 -m 16 -mmax 16 -mmax0 32 -ef 400 -size 128 -csvfile benchmarks/600k-m16-128d.csv
./bin/vecbench -num 700000 -m 16 -mmax 16 -mmax0 32 -ef 400 -size 128 -csvfile benchmarks/700k-m16-128d.csv
./bin/vecbench -num 800000 -m 16 -mmax 16 -mmax0 32 -ef 400 -size 128 -csvfile benchmarks/800k-m16-128d.csv
./bin/vecbench -num 900000 -m 16 -mmax 16 -mmax0 32 -ef 400 -size 128 -csvfile benchmarks/900k-m16-128d.csv
./bin/vecbench -num 1000000 -m 16 -mmax 16 -mmax0 32 -ef 400 -size 128 -csvfile benchmarks/1m-m16-128d.csv