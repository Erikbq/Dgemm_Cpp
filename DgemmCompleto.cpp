#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <string>

using namespace std;

// ============================================================================
// 1. IMPLEMENTAÇÕES DGEMM
// ============================================================================

// Naive (Ingênua - i-j-k)
void dgemm_naive(int n, const double* A, const double* B, double* C) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// Reordered (i-k-j)
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

// Blocked (Tiling)
void dgemm_blocked(int n, const double* A, const double* B, double* C, int blockSize) {
    for (int ii = 0; ii < n; ii += blockSize) {
        for (int jj = 0; jj < n; jj += blockSize) {
            for (int kk = 0; kk < n; kk += blockSize) {
                int iEnd = min(ii + blockSize, n);
                int jEnd = min(jj + blockSize, n);
                int kEnd = min(kk + blockSize, n);
                
                for (int i = ii; i < iEnd; i++) {
                    for (int k = kk; k < kEnd; k++) {
                        double r = A[i * n + k];
                        for (int j = jj; j < jEnd; j++) {
                            C[i * n + j] += r * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

// Unrolled (Desenrolado)
void dgemm_unrolled(int n, const double* A, const double* B, double* C) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double r = A[i * n + k];
            int j = 0;
            for (; j <= n - 4; j += 4) {
                C[i * n + j]     += r * B[k * n + j];
                C[i * n + j + 1] += r * B[k * n + j + 1];
                C[i * n + j + 2] += r * B[k * n + j + 2];
                C[i * n + j + 3] += r * B[k * n + j + 3];
            }
            for (; j < n; j++) {
                C[i * n + j] += r * B[k * n + j];
            }
        }
    }
}

// Blocked + Unrolled (Combinação)
void dgemm_blocked_unrolled(int n, const double* A, const double* B, double* C, int blockSize) {
    for (int ii = 0; ii < n; ii += blockSize) {
        for (int jj = 0; jj < n; jj += blockSize) {
            for (int kk = 0; kk < n; kk += blockSize) {
                int iEnd = min(ii + blockSize, n);
                int jEnd = min(jj + blockSize, n);
                int kEnd = min(kk + blockSize, n);
                
                for (int i = ii; i < iEnd; i++) {
                    for (int k = kk; k < kEnd; k++) {
                        double r = A[i * n + k];
                        int j = jj;
                        for (; j <= jEnd - 4; j += 4) {
                            C[i * n + j]     += r * B[k * n + j];
                            C[i * n + j + 1] += r * B[k * n + j + 1];
                            C[i * n + j + 2] += r * B[k * n + j + 2];
                            C[i * n + j + 3] += r * B[k * n + j + 3];
                        }
                        for (; j < jEnd; j++) {
                            C[i * n + j] += r * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

// Register Blocking (Micro-kernel 4x4)
void dgemm_register_blocking(int n, const double* A, const double* B, double* C) {
    const int MR = 4;
    const int NR = 4;
    
    for (int i = 0; i < n; i += MR) {
        for (int j = 0; j < n; j += NR) {
            double c00 = 0, c01 = 0, c02 = 0, c03 = 0;
            double c10 = 0, c11 = 0, c12 = 0, c13 = 0;
            double c20 = 0, c21 = 0, c22 = 0, c23 = 0;
            double c30 = 0, c31 = 0, c32 = 0, c33 = 0;
            
            if (i < n && j < n) c00 = C[i * n + j];
            if (i < n && j + 1 < n) c01 = C[i * n + j + 1];
            if (i < n && j + 2 < n) c02 = C[i * n + j + 2];
            if (i < n && j + 3 < n) c03 = C[i * n + j + 3];
            
            if (i + 1 < n && j < n) c10 = C[(i + 1) * n + j];
            if (i + 1 < n && j + 1 < n) c11 = C[(i + 1) * n + j + 1];
            if (i + 1 < n && j + 2 < n) c12 = C[(i + 1) * n + j + 2];
            if (i + 1 < n && j + 3 < n) c13 = C[(i + 1) * n + j + 3];
            
            if (i + 2 < n && j < n) c20 = C[(i + 2) * n + j];
            if (i + 2 < n && j + 1 < n) c21 = C[(i + 2) * n + j + 1];
            if (i + 2 < n && j + 2 < n) c22 = C[(i + 2) * n + j + 2];
            if (i + 2 < n && j + 3 < n) c23 = C[(i + 2) * n + j + 3];
            
            if (i + 3 < n && j < n) c30 = C[(i + 3) * n + j];
            if (i + 3 < n && j + 1 < n) c31 = C[(i + 3) * n + j + 1];
            if (i + 3 < n && j + 2 < n) c32 = C[(i + 3) * n + j + 2];
            if (i + 3 < n && j + 3 < n) c33 = C[(i + 3) * n + j + 3];
            
            for (int k = 0; k < n; k++) {
                double a0 = (i < n) ? A[i * n + k] : 0;
                double a1 = (i + 1 < n) ? A[(i + 1) * n + k] : 0;
                double a2 = (i + 2 < n) ? A[(i + 2) * n + k] : 0;
                double a3 = (i + 3 < n) ? A[(i + 3) * n + k] : 0;
                
                double b0 = (j < n) ? B[k * n + j] : 0;
                double b1 = (j + 1 < n) ? B[k * n + j + 1] : 0;
                double b2 = (j + 2 < n) ? B[k * n + j + 2] : 0;
                double b3 = (j + 3 < n) ? B[k * n + j + 3] : 0;
                
                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
            }
            
            if (i < n && j < n) C[i * n + j] = c00;
            if (i < n && j + 1 < n) C[i * n + j + 1] = c01;
            if (i < n && j + 2 < n) C[i * n + j + 2] = c02;
            if (i < n && j + 3 < n) C[i * n + j + 3] = c03;
            
            if (i + 1 < n && j < n) C[(i + 1) * n + j] = c10;
            if (i + 1 < n && j + 1 < n) C[(i + 1) * n + j + 1] = c11;
            if (i + 1 < n && j + 2 < n) C[(i + 1) * n + j + 2] = c12;
            if (i + 1 < n && j + 3 < n) C[(i + 1) * n + j + 3] = c13;
            
            if (i + 2 < n && j < n) C[(i + 2) * n + j] = c20;
            if (i + 2 < n && j + 1 < n) C[(i + 2) * n + j + 1] = c21;
            if (i + 2 < n && j + 2 < n) C[(i + 2) * n + j + 2] = c22;
            if (i + 2 < n && j + 3 < n) C[(i + 2) * n + j + 3] = c23;
            
            if (i + 3 < n && j < n) C[(i + 3) * n + j] = c30;
            if (i + 3 < n && j + 1 < n) C[(i + 3) * n + j + 1] = c31;
            if (i + 3 < n && j + 2 < n) C[(i + 3) * n + j + 2] = c32;
            if (i + 3 < n && j + 3 < n) C[(i + 3) * n + j + 3] = c33;
        }
    }
}

// ============================================================================
// 2. UTILITÁRIOS E BENCHMARK (Baseado no analise_comparativa.js)
// ============================================================================

void init_matrix(int n, double* M, bool isZero) {
    for (int i = 0; i < n * n; ++i) {
        M[i] = isZero ? 0.0 : (double)(rand() % 100) / 10.0;
    }
}

// Estrutura para armazenar resultados do benchmark
struct BenchmarkResult {
    double avgTime;
    double gflops;
};

// Função genérica de benchmark
template<typename Func>
BenchmarkResult run_benchmark(string name, Func func, int n, int iterations) {
    cout << "  Testando " << name << " com n=" << n << "...\n";
    
    double* A = new double[n * n];
    double* B = new double[n * n];
    double* C = new double[n * n];
    
    vector<double> times;
    
    for (int iter = 0; iter < iterations; iter++) {
        init_matrix(n, A, false);
        init_matrix(n, B, false);
        init_matrix(n, C, true);
        
        auto start = chrono::high_resolution_clock::now();
        func(n, A, B, C);
        auto end = chrono::high_resolution_clock::now();
        
        chrono::duration<double> diff = end - start;
        times.push_back(diff.count());
    }
    
    double totalTime = 0;
    for (double t : times) totalTime += t;
    double avgTime = totalTime / iterations;
    
    // GFLOPS = (2 * n^3) / (tempo * 10^9)
    double flops = 2.0 * pow(n, 3);
    double gflops = (flops / avgTime) / 1e9;
    
    delete[] A; delete[] B; delete[] C;
    
    return {avgTime * 1000.0, gflops}; // Retorna tempo em milissegundos
}

// ============================================================================
// 3. MAIN (Execução dos Experimentos)
// ============================================================================

int main() {
    srand(42); // Semente fixa para reproducibilidade
    
    cout << string(80, '=') << "\n";
    cout << "                    ANÁLISE COMPARATIVA COMPLETA - DGEMM\n";
    cout << string(80, '=') << "\n\n";

    vector<int> testSizes = {64, 128, 256, 512};
    int iterations = 3;

    cout << "EXPERIMENTO: COMPARAÇÃO DE TÉCNICAS (Tempo Médio em ms / GFLOPS)\n";
    cout << string(80, '-') << "\n";
    
    // Tabela Header
    cout << left << setw(10) << "Tamanho"
         << "| " << setw(15) << "Naive"
         << "| " << setw(15) << "Reordered"
         << "| " << setw(15) << "Unrolled"
         << "| " << setw(15) << "Blocked (32)" 
         << "| " << setw(15) << "Reg Blocked\n";
    cout << string(80, '-') << "\n";

    for (int size : testSizes) {
        // Wrappers para funções com parâmetros extras
        auto wrap_naive = [](int n, double* A, double* B, double* C) { dgemm_naive(n, A, B, C); };
        auto wrap_reord = [](int n, double* A, double* B, double* C) { dgemm_reordered(n, A, B, C); };
        auto wrap_unrol = [](int n, double* A, double* B, double* C) { dgemm_unrolled(n, A, B, C); };
        auto wrap_block = [](int n, double* A, double* B, double* C) { dgemm_blocked(n, A, B, C, 32); };
        auto wrap_regbk = [](int n, double* A, double* B, double* C) { dgemm_register_blocking(n, A, B, C); };

        auto res_naive = run_benchmark("Naive", wrap_naive, size, iterations);
        auto res_reord = run_benchmark("Reordered", wrap_reord, size, iterations);
        auto res_unrol = run_benchmark("Unrolled", wrap_unrol, size, iterations);
        auto res_block = run_benchmark("Blocked (32)", wrap_block, size, iterations);
        auto res_regbk = run_benchmark("Reg Blocked", wrap_regbk, size, iterations);

        // Imprimindo Resultados
        cout << "\n" << left << setw(10) << (to_string(size) + "x" + to_string(size))
             << "| " << fixed << setprecision(2) << res_naive.avgTime << "ms / " << res_naive.gflops << " GF"
             << "\n" << setw(10) << " " 
             << "| " << fixed << setprecision(2) << res_reord.avgTime << "ms / " << res_reord.gflops << " GF"
             << "\n" << setw(10) << " " 
             << "| " << fixed << setprecision(2) << res_unrol.avgTime << "ms / " << res_unrol.gflops << " GF"
             << "\n" << setw(10) << " " 
             << "| " << fixed << setprecision(2) << res_block.avgTime << "ms / " << res_block.gflops << " GF"
             << "\n" << setw(10) << " " 
             << "| " << fixed << setprecision(2) << res_regbk.avgTime << "ms / " << res_regbk.gflops << " GF\n";
        cout << string(80, '-') << "\n";
    }

    cout << "\nAnálise completa! Note como o GFLOPS aumenta nas matrizes maiores com o uso de Cache Blocking.\n";
    return 0;
}