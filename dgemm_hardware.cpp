#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <immintrin.h>
#include <omp.h>

using namespace std;

// ============================================================================
// Função auxiliar de índice
// Matriz em row-major: M[i][j] = M[i * n + j]
// ============================================================================
inline int IDX(int i, int j, int n) {
    return i * n + j;
}

// ============================================================================
// Capítulo 2: C/C++ básico, versão ingênua i-j-k
// ============================================================================
void dgemm_naive(int n, const double* A, const double* B, double* C) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[IDX(i, j, n)] += A[IDX(i, k, n)] * B[IDX(k, j, n)];
            }
        }
    }
}

// ============================================================================
// Extra: reordenação i-k-j
// Melhora localidade espacial em B e C
// ============================================================================
void dgemm_reordered(int n, const double* A, const double* B, double* C) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double a_val = A[IDX(i, k, n)];

            for (int j = 0; j < n; j++) {
                C[IDX(i, j, n)] += a_val * B[IDX(k, j, n)];
            }
        }
    }
}

// ============================================================================
// Capítulo 3: SIMD com AVX2
// Processa 4 doubles por vez usando registrador de 256 bits
// ============================================================================
void dgemm_simd(int n, const double* A, const double* B, double* C) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            __m256d a_vec = _mm256_set1_pd(A[IDX(i, k, n)]);

            int j = 0;

            for (; j <= n - 4; j += 4) {
                __m256d c_vec = _mm256_loadu_pd(&C[IDX(i, j, n)]);
                __m256d b_vec = _mm256_loadu_pd(&B[IDX(k, j, n)]);

                c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);

                _mm256_storeu_pd(&C[IDX(i, j, n)], c_vec);
            }

            // Resto escalar caso n não seja múltiplo de 4
            for (; j < n; j++) {
                C[IDX(i, j, n)] += A[IDX(i, k, n)] * B[IDX(k, j, n)];
            }
        }
    }
}

// ============================================================================
// Capítulo 4: SIMD + loop unrolling
// Processa 16 doubles por iteração: 4 vetores de 4 doubles
// ============================================================================
void dgemm_unroll_simd(int n, const double* A, const double* B, double* C) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            __m256d a_vec = _mm256_set1_pd(A[IDX(i, k, n)]);

            int j = 0;

            for (; j <= n - 16; j += 16) {
                __m256d c0 = _mm256_loadu_pd(&C[IDX(i, j, n)]);
                __m256d b0 = _mm256_loadu_pd(&B[IDX(k, j, n)]);
                c0 = _mm256_fmadd_pd(a_vec, b0, c0);
                _mm256_storeu_pd(&C[IDX(i, j, n)], c0);

                __m256d c1 = _mm256_loadu_pd(&C[IDX(i, j + 4, n)]);
                __m256d b1 = _mm256_loadu_pd(&B[IDX(k, j + 4, n)]);
                c1 = _mm256_fmadd_pd(a_vec, b1, c1);
                _mm256_storeu_pd(&C[IDX(i, j + 4, n)], c1);

                __m256d c2 = _mm256_loadu_pd(&C[IDX(i, j + 8, n)]);
                __m256d b2 = _mm256_loadu_pd(&B[IDX(k, j + 8, n)]);
                c2 = _mm256_fmadd_pd(a_vec, b2, c2);
                _mm256_storeu_pd(&C[IDX(i, j + 8, n)], c2);

                __m256d c3 = _mm256_loadu_pd(&C[IDX(i, j + 12, n)]);
                __m256d b3 = _mm256_loadu_pd(&B[IDX(k, j + 12, n)]);
                c3 = _mm256_fmadd_pd(a_vec, b3, c3);
                _mm256_storeu_pd(&C[IDX(i, j + 12, n)], c3);
            }

            // Continua com SIMD para o que ainda couber em blocos de 4
            for (; j <= n - 4; j += 4) {
                __m256d c_vec = _mm256_loadu_pd(&C[IDX(i, j, n)]);
                __m256d b_vec = _mm256_loadu_pd(&B[IDX(k, j, n)]);

                c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);

                _mm256_storeu_pd(&C[IDX(i, j, n)], c_vec);
            }

            // Resto escalar
            for (; j < n; j++) {
                C[IDX(i, j, n)] += A[IDX(i, k, n)] * B[IDX(k, j, n)];
            }
        }
    }
}

// ============================================================================
// Capítulo 5: Cache blocking + SIMD
// Ainda sem OpenMP
// ============================================================================
void dgemm_blocked(int n, const double* A, const double* B, double* C) {
    const int BLOCKSIZE = 32;

    for (int ii = 0; ii < n; ii += BLOCKSIZE) {
        for (int jj = 0; jj < n; jj += BLOCKSIZE) {
            for (int kk = 0; kk < n; kk += BLOCKSIZE) {

                int iEnd = min(ii + BLOCKSIZE, n);
                int jEnd = min(jj + BLOCKSIZE, n);
                int kEnd = min(kk + BLOCKSIZE, n);

                for (int i = ii; i < iEnd; i++) {
                    for (int k = kk; k < kEnd; k++) {
                        __m256d a_vec = _mm256_set1_pd(A[IDX(i, k, n)]);

                        int j = jj;

                        for (; j <= jEnd - 4; j += 4) {
                            __m256d c_vec = _mm256_loadu_pd(&C[IDX(i, j, n)]);
                            __m256d b_vec = _mm256_loadu_pd(&B[IDX(k, j, n)]);

                            c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);

                            _mm256_storeu_pd(&C[IDX(i, j, n)], c_vec);
                        }

                        for (; j < jEnd; j++) {
                            C[IDX(i, j, n)] += A[IDX(i, k, n)] * B[IDX(k, j, n)];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Capítulo 6: Cache blocking + SIMD + OpenMP
// Paraleliza os blocos de C
// ============================================================================
void dgemm_blocked_omp(int n, const double* A, const double* B, double* C) {
    const int BLOCKSIZE = 32;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += BLOCKSIZE) {
        for (int jj = 0; jj < n; jj += BLOCKSIZE) {
            for (int kk = 0; kk < n; kk += BLOCKSIZE) {

                int iEnd = min(ii + BLOCKSIZE, n);
                int jEnd = min(jj + BLOCKSIZE, n);
                int kEnd = min(kk + BLOCKSIZE, n);

                for (int i = ii; i < iEnd; i++) {
                    for (int k = kk; k < kEnd; k++) {
                        __m256d a_vec = _mm256_set1_pd(A[IDX(i, k, n)]);

                        int j = jj;

                        for (; j <= jEnd - 4; j += 4) {
                            __m256d c_vec = _mm256_loadu_pd(&C[IDX(i, j, n)]);
                            __m256d b_vec = _mm256_loadu_pd(&B[IDX(k, j, n)]);

                            c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);

                            _mm256_storeu_pd(&C[IDX(i, j, n)], c_vec);
                        }

                        for (; j < jEnd; j++) {
                            C[IDX(i, j, n)] += A[IDX(i, k, n)] * B[IDX(k, j, n)];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Inicialização das matrizes
// ============================================================================
void init_matrix(int n, double* M, bool zero) {
    size_t total = (size_t)n * (size_t)n;

    for (size_t i = 0; i < total; i++) {
        if (zero) {
            M[i] = 0.0;
        } else {
            M[i] = (double)((i * 13 + 7) % 100) / 10.0;
        }
    }
}

// ============================================================================
// Calcula erro máximo entre duas matrizes
// ============================================================================
double max_error(int n, const double* X, const double* Y) {
    size_t total = (size_t)n * (size_t)n;
    double erro = 0.0;

    for (size_t i = 0; i < total; i++) {
        erro = max(erro, fabs(X[i] - Y[i]));
    }

    return erro;
}

// ============================================================================
// Benchmark genérico
// ============================================================================
template <typename Func>
double run_benchmark(
    const string& name,
    Func func,
    int n,
    const double* A,
    const double* B,
    double* C,
    const double* reference,
    bool validate
) {
    init_matrix(n, C, true);

    auto start = chrono::high_resolution_clock::now();

    func(n, A, B, C);

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> diff = end - start;
    double seconds = diff.count();

    double flops = 2.0 * n * n * n;
    double gflops = flops / seconds / 1e9;

    cout << left << setw(35) << name
         << "| Tempo: " << fixed << setprecision(6) << seconds << " s "
         << "| GFLOPS: " << fixed << setprecision(3) << gflops;

    if (validate && reference != nullptr) {
        double erro = max_error(n, C, reference);
        bool ok = erro <= 1e-7;

        cout << " | Erro max: " << scientific << setprecision(3) << erro
             << " | " << (ok ? "OK" : "ERRO");

        cout << fixed;
    } else {
        cout << " | Referencia";
    }

    cout << "\n";

    return seconds;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    vector<int> sizes = {512, 1024, 2048};

    cout << "===============================================================\n";
    cout << "        DGEMM - Going Faster - Capitulos 2 a 6\n";
    cout << "===============================================================\n";
    cout << "Threads maximas OpenMP: " << omp_get_max_threads() << "\n";

    for (int n : sizes) {
        cout << "\n>>> Matriz: " << n << " x " << n << " <<<\n";

        size_t total = (size_t)n * (size_t)n;

        vector<double> A(total);
        vector<double> B(total);
        vector<double> C(total);
        vector<double> C_ref(total);

        init_matrix(n, A.data(), false);
        init_matrix(n, B.data(), false);

        // Capítulo 2 como referência correta
        run_benchmark(
            "Cap. 2: C++ naive i-j-k",
            dgemm_naive,
            n,
            A.data(),
            B.data(),
            C.data(),
            nullptr,
            false
        );

        C_ref = C;

        // Outras versões comparadas contra a referência
        run_benchmark(
            "Extra: reordered i-k-j",
            dgemm_reordered,
            n,
            A.data(),
            B.data(),
            C.data(),
            C_ref.data(),
            true
        );

        run_benchmark(
            "Cap. 3: SIMD AVX2",
            dgemm_simd,
            n,
            A.data(),
            B.data(),
            C.data(),
            C_ref.data(),
            true
        );

        run_benchmark(
            "Cap. 4: SIMD + unrolling",
            dgemm_unroll_simd,
            n,
            A.data(),
            B.data(),
            C.data(),
            C_ref.data(),
            true
        );

        run_benchmark(
            "Cap. 5: blocking + SIMD",
            dgemm_blocked,
            n,
            A.data(),
            B.data(),
            C.data(),
            C_ref.data(),
            true
        );

        run_benchmark(
            "Cap. 6: blocking + SIMD + OpenMP",
            dgemm_blocked_omp,
            n,
            A.data(),
            B.data(),
            C.data(),
            C_ref.data(),
            true
        );
    }

    return 0;
}