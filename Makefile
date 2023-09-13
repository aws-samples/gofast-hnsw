GO_PROJECT_NAME := gofast-hnsw

# GO commands
go_build:
	@echo "Building $(GO_PROJECT_NAME)"
	go build -pgo=off -o bin/vecbench-no_pgo cmd/vecbench/main.go
	go build -o bin/vecbench cmd/vecbench/main.go

go_pgo:
	@echo "\n....Running $(GO_PROJECT_NAME) for CPU profile ...."
	./bin/vecbench -profile cmd/vecbench/default.pgo -groundtruth=false -num 50000 -size 64 -m 16 -mmax 16 -mmax0 32 -ef 100

go_run:
	@echo "\n....Running $(GO_PROJECT_NAME)...."

test:
	@echo "\n....Running tests for $(GO_PROJECT_NAME)...."
	LOG_IGNORE=1 go test -v ./...

bench:
	@echo "\n....Running benchmarks for $(GO_PROJECT_NAME)...."
	LOG_IGNORE=1 go test -benchmem -run=. -bench=. ./...

# Project rules
pgo:
	$(MAKE) go_build
	$(MAKE) go_pgo
	$(MAKE) go_build

build:
	$(MAKE) go_build

run:
	$(MAKE) go_build
	$(MAKE) go_run

clean:
	rm -f ./bin/*

.PHONY: go_build go_run build run test
