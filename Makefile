all: performance_host

performance_host: performance_host.cpp
	g++ -O3 -march=native -std=c++11 -fopenmp performance_host.cpp -o performance_host

clean:
	rm -rf performance_host
