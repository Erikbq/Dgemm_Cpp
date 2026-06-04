CXX = g++

# Adicionamos o -mavx para liberar instruções SIMD e o -fopenmp para Multithreading
CXXFLAGS = -O3 -march=native -mavx -fopenmp -Wall -std=c++11

TARGET = dgemm_hardware

all: $(TARGET)

$(TARGET): dgemm_hardware.cpp
	$(CXX) $(CXXFLAGS) dgemm_hardware.cpp -o $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)