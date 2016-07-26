all: performance_host correctness_host

performance_host: performance_host.cpp
	g++ -O3 -march=native -std=c++11 -fopenmp performance_host.cpp -o performance_host

correctness_host: correctness_host.cpp
	g++ -O3 -march=native -std=c++11 -fopenmp correctness_host.cpp -o correctness_host

clean:
	rm -rf performance_host correctness_host
