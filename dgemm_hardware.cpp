#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>   // std::min
#include <immintrin.h> // SIMD AVX
#include <omp.h>       // OpenMP

using namespace std;

// ============================================================================
// BASELINE (Ingênuo - Sem Otimização)
// ============================================================================
void dgemm_naive(int n, const double* A, const double* B, double* C) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// ============================================================================
// CAPÍTULO 3: Subword Parallelism (SIMD AVX)
// ============================================================================
void dgemm_ch3_simd(int n, const double* A, const double* B, double* C) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            __m256d a_val = _mm256_set1_pd(A[i * n + k]);
            
            for (int j = 0; j < n; j += 4) {
                // Prevenção contra falha de segmentação se 'n' não for múltiplo de 4
                if (j + 3 < n) {
                    __m256d c_vec = _mm256_loadu_pd(&C[i * n + j]);
                    __m256d b_vec = _mm256_loadu_pd(&B[k * n + j]);
                    c_vec = _mm256_add_pd(c_vec, _mm256_mul_pd(a_val, b_vec));
                    _mm256_storeu_pd(&C[i * n + j], c_vec);
                } else {
                    // Laço de sobra (Fringe Loop)
                    for (int rem = j; rem < n; rem++) {
                        C[i * n + rem] += A[i * n + k] * B[k * n + rem];
                    }
                }
            }
        }
    }
}

// ============================================================================
// CAPÍTULO 4: Instruction-Level Parallelism (SIMD + Loop Unrolling)
// ============================================================================
void dgemm_ch4_unroll(int n, const double* A, const double* B, double* C) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            __m256d a_val = _mm256_set1_pd(A[i * n + k]);
            
            for (int j = 0; j < n; j += 16) {
                if (j + 15 < n) {
                    _mm256_storeu_pd(&C[i*n+j],    _mm256_add_pd(_mm256_loadu_pd(&C[i*n+j]),    _mm256_mul_pd(a_val, _mm256_loadu_pd(&B[k*n+j]))));
                    _mm256_storeu_pd(&C[i*n+j+4],  _mm256_add_pd(_mm256_loadu_pd(&C[i*n+j+4]),  _mm256_mul_pd(a_val, _mm256_loadu_pd(&B[k*n+j+4]))));
                    _mm256_storeu_pd(&C[i*n+j+8],  _mm256_add_pd(_mm256_loadu_pd(&C[i*n+j+8]),  _mm256_mul_pd(a_val, _mm256_loadu_pd(&B[k*n+j+8]))));
                    _mm256_storeu_pd(&C[i*n+j+12], _mm256_add_pd(_mm256_loadu_pd(&C[i*n+j+12]), _mm256_mul_pd(a_val, _mm256_loadu_pd(&B[k*n+j+12]))));
                } else {
                    for (int rem = j; rem < n; rem++) {
                        C[i * n + rem] += A[i * n + k] * B[k * n + rem];
                    }
                }
            }
        }
    }
}

// ============================================================================
// CAPÍTULO 5: Hierarquia de Memória (SIMD + Unroll + Cache Blocking)
// ============================================================================
void dgemm_ch5_block(int n, const double* A, const double* B, double* C) {
    const int BLOCKSIZE = 32;
    for (int ii = 0; ii < n; ii += BLOCKSIZE) {
        for (int jj = 0; jj < n; jj += BLOCKSIZE) {
            for (int kk = 0; kk < n; kk += BLOCKSIZE) {
                
                // Cálculo de borda seguro para matrizes de tamanhos arbitrários
                int iEnd = min(ii + BLOCKSIZE, n);
                int jEnd = min(jj + BLOCKSIZE, n);
                int kEnd = min(kk + BLOCKSIZE, n);
                
                for (int i = ii; i < iEnd; i++) {
                    for (int k = kk; k < kEnd; k++) {
                        __m256d a_val = _mm256_set1_pd(A[i * n + k]);
                        
                        for (int j = jj; j < jEnd; j += 16) {
                            if (j + 15 < jEnd) {
                                _mm256_storeu_pd(&C[i*n+j],    _mm256_add_pd(_mm256_loadu_pd(&C[i*n+j]),    _mm256_mul_pd(a_val, _mm256_loadu_pd(&B[k*n+j]))));
                                _mm256_storeu_pd(&C[i*n+j+4],  _mm256_add_pd(_mm256_loadu_pd(&C[i*n+j+4]),  _mm256_mul_pd(a_val, _mm256_loadu_pd(&B[k*n+j+4]))));
                                _mm256_storeu_pd(&C[i*n+j+8],  _mm256_add_pd(_mm256_loadu_pd(&C[i*n+j+8]),  _mm256_mul_pd(a_val, _mm256_loadu_pd(&B[k*n+j+8]))));
                                _mm256_storeu_pd(&C[i*n+j+12], _mm256_add_pd(_mm256_loadu_pd(&C[i*n+j+12]), _mm256_mul_pd(a_val, _mm256_loadu_pd(&B[k*n+j+12]))));
                            } else {
                                for (int rem = j; rem < jEnd; rem++) {
                                    C[i * n + rem] += A[i * n + k] * B[k * n + rem];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// CAPÍTULO 6: Processadores Paralelos (SIMD + Unroll + Block + OpenMP)
// ============================================================================
void dgemm_ch6_omp(int n, const double* A, const double* B, double* C) {
    const int BLOCKSIZE = 32;
    
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < n; ii += BLOCKSIZE) {
        for (int jj = 0; jj < n; jj += BLOCKSIZE) {
            for (int kk = 0; kk < n; kk += BLOCKSIZE) {
                
                int iEnd = min(ii + BLOCKSIZE, n);
                int jEnd = min(jj + BLOCKSIZE, n);
                int kEnd = min(kk + BLOCKSIZE, n);
                
                for (int i = ii; i < iEnd; i++) {
                    for (int k = kk; k < kEnd; k++) {
                        __m256d a_val = _mm256_set1_pd(A[i * n + k]);
                        
                        for (int j = jj; j < jEnd; j += 16) {
                            if (j + 15 < jEnd) {
                                _mm256_storeu_pd(&C[i*n+j],    _mm256_add_pd(_mm256_loadu_pd(&C[i*n+j]),    _mm256_mul_pd(a_val, _mm256_loadu_pd(&B[k*n+j]))));
                                _mm256_storeu_pd(&C[i*n+j+4],  _mm256_add_pd(_mm256_loadu_pd(&C[i*n+j+4]),  _mm256_mul_pd(a_val, _mm256_loadu_pd(&B[k*n+j+4]))));
                                _mm256_storeu_pd(&C[i*n+j+8],  _mm256_add_pd(_mm256_loadu_pd(&C[i*n+j+8]),  _mm256_mul_pd(a_val, _mm256_loadu_pd(&B[k*n+j+8]))));
                                _mm256_storeu_pd(&C[i*n+j+12], _mm256_add_pd(_mm256_loadu_pd(&C[i*n+j+12]), _mm256_mul_pd(a_val, _mm256_loadu_pd(&B[k*n+j+12]))));
                            } else {
                                for (int rem = j; rem < jEnd; rem++) {
                                    C[i * n + rem] += A[i * n + k] * B[k * n + rem];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// FUNÇÕES UTILITÁRIAS E BENCHMARK ROBUSTO
// ============================================================================
void init_matrix(int n, double* M, bool isZero) {
    for (int i = 0; i < n * n; ++i) {
        M[i] = isZero ? 0.0 : (double)(rand() % 100) / 10.0;
    }
}

template<typename Func>
void run_benchmark(string name, Func func, int n, double* A, double* B, double* C, int iterations = 3) {
    double total_seconds = 0.0;
    
    for (int iter = 0; iter < iterations; iter++) {
        // Zera o resultado antes de cada execução para garantir cálculos justos
        init_matrix(n, C, true); 
        
        auto start = chrono::high_resolution_clock::now();
        func(n, A, B, C);
        auto end = chrono::high_resolution_clock::now();
        
        chrono::duration<double> diff = end - start;
        total_seconds += diff.count();
    }
    
    // Média de tempo e cálculo dos GFLOPS
    double avg_seconds = total_seconds / iterations;
    double gflops = (2.0 * n * n * n) / avg_seconds / 1e9;
    
    cout << left << setw(35) << name 
         << "| Tempo Medio (" << iterations << "x): " << fixed << setprecision(4) << avg_seconds << " s "
         << "| GFLOPS: " << fixed << setprecision(2) << gflops << "\n";
}

int main() {
    vector<int> sizes = {256, 512, 1024, 2048}; 

    cout << "=========================================================\n";
    cout << "  BATTERY OF TESTS: DGEMM (3 ITERAÇÕES POR MODELO)\n";
    cout << "=========================================================\n";

    for (int n : sizes) {
        cout << "\n>>> Testando matriz de tamanho: " << n << "x" << n << " <<<\n";
        
        double* A = new double[n * n];
        double* B = new double[n * n];
        double* C = new double[n * n];

        init_matrix(n, A, false);
        init_matrix(n, B, false);

        run_benchmark("1. Baseline (Sem Otimizacao)", dgemm_naive, n, A, B, C, 3);
        run_benchmark("2. Cap. 3: SIMD AVX", dgemm_ch3_simd, n, A, B, C, 3);
        run_benchmark("3. Cap. 4: SIMD + Unroll", dgemm_ch4_unroll, n, A, B, C, 3);
        run_benchmark("4. Cap. 5: SIMD + Unroll + Block", dgemm_ch5_block, n, A, B, C, 3);
        run_benchmark("5. Cap. 6: Block + OMP (ALL)", dgemm_ch6_omp, n, A, B, C, 3);

        delete[] A; delete[] B; delete[] C;
    }

    return 0;
}