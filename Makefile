simd:
	g++ -std=c++11 -O3 -fopenmp -mavx -mfma main.cpp -o ELM
core:
	g++ -std=c++11 -O3 -fopenmp main.cpp -o ELM
