// Compilação:
// mpicc -O3 -Ofast -fopenmp -march=native -funroll-loops mpi.c -o guirafa.exe
//
// Execução (exemplo):
// mpiexec -hostfile hosts -np 8 --bind-to core --map-by ppr:1:socket ./guirafa

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef NUMERO
#define NUMERO 4000
#endif

#ifndef TILE
#define TILE 128
#endif

int _tile = TILE;
int _matrix_length = NUMERO;

static inline int min_i(int a, int b) { return a < b ? a : b; }

static int pick_tile(void) {
    return _tile;
}

void get_args(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) 
    {
        if ((strcmp(argv[i], "--length") == 0 || strcmp(argv[i], "--l") == 0) && i + 1 < argc) {
            _matrix_length = atoi(argv[i + 1]);
            i++;
        } else if ((strcmp(argv[i], "--tile") == 0 || strcmp(argv[i], "--t") == 0) && i + 1 < argc) {
            _tile = atoi(argv[i + 1]);
            i++;
        }
    }
}

int main(int argc, char** argv) {

    get_args(argc, argv);

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = _matrix_length;
    const int T = pick_tile();

    // Distribuição de linhas (não requer N % size == 0)
    int *counts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));
    if (!counts || !displs) {
        if (rank == 0) fprintf(stderr, "Falha malloc counts/displs\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int base  = N / size;
    const int extra = N % size;
    for (int r = 0; r < size; ++r) {
        int rows = base + (r < extra ? 1 : 0);
        counts[r] = rows; // em linhas
    }
    displs[0] = 0;
    for (int r = 1; r < size; ++r) {
        displs[r] = displs[r - 1] + counts[r - 1];
    }

    const int rows_local = counts[rank];
    const size_t bytesA_local = (size_t)rows_local * N * sizeof(int32_t);
    const size_t bytesB       = (size_t)N * N * sizeof(int32_t);
    const size_t bytesC_local = (size_t)rows_local * N * sizeof(int32_t);

    int32_t *A_root = NULL;                 // apenas rank 0
    int32_t *B      = (int32_t*)malloc(bytesB);
    int32_t *BT     = (int32_t*)malloc(bytesB);      // B transposta
    int32_t *A_loc  = (int32_t*)malloc(bytesA_local);
    int32_t *C_loc  = (int32_t*)malloc(bytesC_local);

    if (!B || !BT || (!A_loc && rows_local > 0) || (!C_loc && rows_local > 0)) {
        fprintf(stderr, "[%d] Falha malloc (B/BT/A_loc/C_loc)\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Rank 0 gera dados
    if (rank == 0) {
        A_root = (int32_t*)malloc((size_t)N * N * sizeof(int32_t));
        if (!A_root) {
            fprintf(stderr, "Falha malloc A_root.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        srand(12345u);
        for (int i = 0; i < N * N; ++i) {
            A_root[i] = rand() % 2;
            B[i]      = rand() % 2;
        }
    }

    // Scatterv usa counts/displs em elementos, não em linhas
    int *sendcounts_elems = (int*)malloc(size * sizeof(int));
    int *displs_elems     = (int*)malloc(size * sizeof(int));
    if (!sendcounts_elems || !displs_elems) {
        if (rank == 0) fprintf(stderr, "Falha malloc sendcounts/displs elems\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int r = 0; r < size; ++r) {
        sendcounts_elems[r] = counts[r] * N;
        displs_elems[r]     = displs[r] * N;
    }

    // Sincroniza todo mundo antes de começar a medir:
    // o tempo medido inclui Scatterv, Broadcast, transposição, multiplicação e Gatherv.
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Scatterv das linhas de A
    MPI_Scatterv(A_root, sendcounts_elems, displs_elems, MPI_INT32_T,
                 A_loc, rows_local * N, MPI_INT32_T,
                 0, MPI_COMM_WORLD);

    // Broadcast de B completa
    MPI_Bcast(B, N * N, MPI_INT32_T, 0, MPI_COMM_WORLD);

    // Transposição local de B -> BT (row-major)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            BT[(size_t)j * N + i] = B[(size_t)i * N + j];
        }
    }

    // Zera C_local
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows_local * N; ++i) {
        C_loc[i] = 0;
    }

    // Multiplicação bloqueada: somente nas linhas deste rank
    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < rows_local; ii += T) {
        for (int jj = 0; jj < N; jj += T) {
            for (int kk = 0; kk < N; kk += T) {

                const int i_max = min_i(ii + T, rows_local);
                const int j_max = min_i(jj + T, N);
                const int k_max = min_i(kk + T, N);

                for (int i = ii; i < i_max; ++i) {
                    int32_t *restrict Ci =
                        &C_loc[(size_t)i * N + jj];
                    const int32_t *restrict Ai =
                        &A_loc[(size_t)i * N + kk];

                    for (int j = jj; j < j_max; ++j) {
                        const int32_t *restrict BTj =
                            &BT[(size_t)j * N + kk];
                        int sum = 0;

                        #pragma omp simd reduction(+:sum)
                        for (int k = kk; k < k_max; ++k) {
                            sum += Ai[k - kk] * BTj[k - kk];
                        }
                        Ci[j - jj] += sum;
                    }
                }
            }
        }
    }

    // Reúne C no rank 0
    int32_t *C_root = NULL;
    if (rank == 0) {
        C_root = (int32_t*)malloc((size_t)N * N * sizeof(int32_t));
        if (!C_root) {
            fprintf(stderr, "Falha malloc C_root.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gatherv(C_loc, rows_local * N, MPI_INT32_T,
                C_root, sendcounts_elems, displs_elems, MPI_INT32_T,
                0, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();
    double mytime  = t1 - t0;
    double maxtime = 0.0;

    // Tempo de parede = maior tempo entre todos os ranks
    MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long long checksum = 0;
        for (int i = 0; i < N * N; ++i) {
            checksum += C_root[i];
        }
        printf("N=%d TILE=%d  tempo_total=%.3f s  checksum=%lld\n",
               N, T, maxtime, checksum);
    }

    // Libera memória
    if (rank == 0) {
        free(A_root);
        free(C_root);
    }
    free(B);
    free(BT);
    free(A_loc);
    free(C_loc);
    free(counts);
    free(displs);
    free(sendcounts_elems);
    free(displs_elems);

    MPI_Finalize();
    return 0;
}
