#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <immintrin.h> // Essencial: Habilita as instruções vetoriais SIMD (AVX) do processador
#include <omp.h>       // Essencial: Habilita o paralelismo de múltiplos núcleos

using namespace std;

// ============================================================================
// 1. BASELINE (Ingênuo - i-j-k)
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
// 2. REORDERED (i-k-j) - Localidade Espacial
// ============================================================================
void dgemm_reordered(int n, const double* A, const double* B, double* C) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double r = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += r * B[k * n + j];
            }
        }
    }
}

// ============================================================================
// 3. SIMD (Subword Parallelism com AVX)
// ============================================================================
// Processa 4 doubles de uma vez usando registradores de 256 bits
void dgemm_simd(int n, const double* A, const double* B, double* C) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            // Carrega A[i][k] para todas as 4 posições do registrador vetorial
            __m256d a_val = _mm256_set1_pd(A[i * n + k]);
            
            for (int j = 0; j < n; j += 4) {
                // Carrega 4 elementos de C e 4 de B
                __m256d c_vec = _mm256_loadu_pd(&C[i * n + j]);
                __m256d b_vec = _mm256_loadu_pd(&B[k * n + j]);
                
                // C = C + (A * B) em um único ciclo
                c_vec = _mm256_add_pd(c_vec, _mm256_mul_pd(a_val, b_vec));
                
                // Guarda o resultado de volta em C
                _mm256_storeu_pd(&C[i * n + j], c_vec);
            }
        }
    }
}

// ============================================================================
// 4. SIMD + LOOP UNROLLING (Nível de Instrução)
// ============================================================================
void dgemm_unroll_simd(int n, const double* A, const double* B, double* C) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            __m256d a_val = _mm256_set1_pd(A[i * n + k]);
            
            // Desenrola processando 16 elementos (4 blocos vetoriais) por iteração
            for (int j = 0; j < n; j += 16) {
                __m256d c0 = _mm256_loadu_pd(&C[i * n + j]);
                __m256d b0 = _mm256_loadu_pd(&B[k * n + j]);
                _mm256_storeu_pd(&C[i * n + j], _mm256_add_pd(c0, _mm256_mul_pd(a_val, b0)));

                __m256d c1 = _mm256_loadu_pd(&C[i * n + j + 4]);
                __m256d b1 = _mm256_loadu_pd(&B[k * n + j + 4]);
                _mm256_storeu_pd(&C[i * n + j + 4], _mm256_add_pd(c1, _mm256_mul_pd(a_val, b1)));

                __m256d c2 = _mm256_loadu_pd(&C[i * n + j + 8]);
                __m256d b2 = _mm256_loadu_pd(&B[k * n + j + 8]);
                _mm256_storeu_pd(&C[i * n + j + 8], _mm256_add_pd(c2, _mm256_mul_pd(a_val, b2)));

                __m256d c3 = _mm256_loadu_pd(&C[i * n + j + 12]);
                __m256d b3 = _mm256_loadu_pd(&B[k * n + j + 12]);
                _mm256_storeu_pd(&C[i * n + j + 12], _mm256_add_pd(c3, _mm256_mul_pd(a_val, b3)));
            }
        }
    }
}

// ============================================================================
// 5. CACHE BLOCKING + SIMD + OPENMP (Desempenho Máximo)
// ============================================================================
void dgemm_final(int n, const double* A, const double* B, double* C) {
    const int BLOCKSIZE = 32;
    
    // Diretiva OpenMP para dividir os blocos entre os múltiplos núcleos da CPU
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
                        for (int j = jj; j < jEnd; j += 4) {
                            __m256d c_vec = _mm256_loadu_pd(&C[i * n + j]);
                            __m256d b_vec = _mm256_loadu_pd(&B[k * n + j]);
                            c_vec = _mm256_add_pd(c_vec, _mm256_mul_pd(a_val, b_vec));
                            _mm256_storeu_pd(&C[i * n + j], c_vec);
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// FUNÇÕES UTILITÁRIAS E BENCHMARK
// ============================================================================
void init_matrix(int n, double* M, bool isZero) {
    for (int i = 0; i < n * n; ++i) {
        M[i] = isZero ? 0.0 : (double)(rand() % 100) / 10.0;
    }
}

template<typename Func>
void run_benchmark(string name, Func func, int n, double* A, double* B, double* C) {
    init_matrix(n, C, true); 
    
    auto start = chrono::high_resolution_clock::now();
    func(n, A, B, C);
    auto end = chrono::high_resolution_clock::now();
    
    chrono::duration<double> diff = end - start;
    double seconds = diff.count();
    double gflops = (2.0 * n * n * n) / seconds / 1e9;
    
    cout << left << setw(25) << name 
         << "| Tempo: " << fixed << setprecision(4) << seconds << " s "
         << "| GFLOPS: " << fixed << setprecision(2) << gflops << "\n";
}

int main() {
    // Vetor com os tamanhos das matrizes que vamos testar
    vector<int> sizes = {256, 512, 1024}; 

    cout << "=========================================================\n";
    cout << "      BATTERY OF TESTS: DGEMM (GOING FASTER)\n";
    cout << "=========================================================\n";

    for (int n : sizes) {
        cout << "\n>>> Testando matriz de tamanho: " << n << "x" << n << " <<<\n";
        
        double* A = new double[n * n];
        double* B = new double[n * n];
        double* C = new double[n * n];

        init_matrix(n, A, false);
        init_matrix(n, B, false);

        run_benchmark("1. Cap. X: Naive (Baseline)", dgemm_naive, n, A, B, C);
        run_benchmark("2. Cap. 5: Reordered", dgemm_reordered, n, A, B, C);
        run_benchmark("3. Cap. 3: SIMD (AVX)", dgemm_simd, n, A, B, C);
        run_benchmark("4. Cap. 4: SIMD + Unroll", dgemm_unroll_simd, n, A, B, C);
        run_benchmark("5. Cap. 6: SIMD + Block + OMP", dgemm_final, n, A, B, C);

        delete[] A; delete[] B; delete[] C;
    }

    return 0;
}