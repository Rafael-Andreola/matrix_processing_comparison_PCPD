// ====== COMENTÁRIOS DE COMPILAÇÃO E EXECUÇÃO ======
// Compilação:
// mpicc -O3 -Ofast -fopenmp -march=native -funroll-loops mpi.c -o mpi.exe
//
// Execução (exemplo):
// mpiexec -hostfile hosts -np 8 --bind-to core --map-by ppr:1:socket ./mpi.exe --length 4000
//
// ====== INCLUSÕES DE BIBLIOTECAS ======
#include <stdio.h>      // Entrada/saída
#include <stdlib.h>     // Funções gerais
#include <stdint.h>     // Tipos inteiros com tamanho fixo (int32_t)
#include <time.h>       // Cronometragem
#include <string.h>     // Funções de string
#include <mpi.h>        // Message Passing Interface (paralelismo distribuído)
#ifdef _OPENMP
#include <omp.h>        // OpenMP para paralelismo em memória compartilhada (híbrido)
#endif

// ====== CONSTANTES ======
// Tamanho padrão da matriz
#ifndef NUMERO
#define NUMERO 4000
#endif

// Tamanho do tile para otimização de cache (hërido MPI+OpenMP)
#ifndef TILE
#define TILE 128
#endif

// ====== VARIÁVEIS GLOBAIS ======
int _tile = TILE;            // Tamanho do tile (pode ser alterado por argumentos)
int _matrix_length = NUMERO;  // Tamanho da matriz

// ====== FUNÇÃO AUX: Mínimo ======
static inline int min_i(int a, int b) { return a < b ? a : b; }

// ====== FUNÇÃO: Retorna Tile Selecionado ======
// DESCRIÇÃO: Simples wrapper para retornar o tamanho do tile
static int pick_tile(void) {
    return _tile;
}

// ====== FUNÇÃO: Processar Argumentos de Linha de Comando ======
// DESCRIÇÃO: Faz parsing dos argumentos fornecidos
// Argumentos suportados:
//   --length, --l: Tamanho da matriz
//   --tile, --t: Tamanho do tile para otimização
void get_args(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) 
    {
        // Processa --length ou --l
        if ((strcmp(argv[i], "--length") == 0 || strcmp(argv[i], "--l") == 0) && i + 1 < argc) {
            _matrix_length = atoi(argv[i + 1]);
            i++;
        } 
        // Processa --tile ou --t
        else if ((strcmp(argv[i], "--tile") == 0 || strcmp(argv[i], "--t") == 0) && i + 1 < argc) {
            _tile = atoi(argv[i + 1]);
            i++;
        }
    }
}

// ====== FUNÇÃO: Main (Ponto de Entrada) ======
// DESCRIÇÃO: Coordena cálculo distribuído usando MPI
// Estratégia: Dividir as linhas de A entre os processos MPI
//            Cada processo calcula suas linhas, depois gather no rank 0
// Paralelismo Híbrído: MPI (entre máquinas) + OpenMP (dentro de cada máquina)
int main(int argc, char** argv) {

    // Parseia argumentos de linha de comando
    get_args(argc, argv);

    // ===== INICIALIZAÇÃO MPI =====
    // MPI_Init() inicializa o ambiente MPI
    // Modifica argc/argv para remover parâmetros do MPI
    MPI_Init(&argc, &argv);

    // ===== OBTEM INFORMAÇÕES DO RANK E TAMANHO =====
    int rank, size;  // rank: ID do processo (0 a size-1), size: número de processos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Qual sou eu?
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Quantos somos?

    // ===== VARIÁVEIS LOCAIS =====
    const int N = _matrix_length;
    const int T = pick_tile();

    // ===== DISTRIBUIÇÃO DE LINHAS =====
    // Estratégia de load balancing: dividir N linhas entre size processos
    // Não requer N % size == 0; distribui linhas extras aos primeiros processos
    // Exemplo: N=10, size=3 => linhas: [4, 3, 3]
    
    // Aloca arrays para contar e deslocar linhas
    int *counts = (int*)malloc(size * sizeof(int));  // Quantas linhas cada processo
    int *displs = (int*)malloc(size * sizeof(int));  // Óndice de cada processo
    if (!counts || !displs) {
        if (rank == 0) fprintf(stderr, "Falha malloc counts/displs\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Calcula distribuição de linhas
    const int base  = N / size;   // Linhas base para cada processo
    const int extra = N % size;   // Linhas extras a distribuir
    for (int r = 0; r < size; ++r) {
        // Primeiros 'extra' processos ganham uma linha extra
        int rows = base + (r < extra ? 1 : 0);
        counts[r] = rows;  // Armazena em linhas
    }
    // Calcula deslocamentos (cumsum dos counts)
    displs[0] = 0;
    for (int r = 1; r < size; ++r) {
        displs[r] = displs[r - 1] + counts[r - 1];
    }

    // Quantas linhas este rank vai processar?
    const int rows_local = counts[rank];
    
    // Tamanhos em bytes para alocação
    const size_t bytesA_local = (size_t)rows_local * N * sizeof(int32_t);  // Linhas locais de A
    const size_t bytesB       = (size_t)N * N * sizeof(int32_t);           // B completa
    const size_t bytesC_local = (size_t)rows_local * N * sizeof(int32_t);  // Linhas locais de C

    // ===== ALOCAÇÃO DE MEMÓRIA =====
    int32_t *A_root = NULL;                  // só rank 0 aloca isso
    int32_t *B      = (int32_t*)malloc(bytesB);           // B completa (broadcast)
    int32_t *BT     = (int32_t*)malloc(bytesB);           // B transposta
    int32_t *A_loc  = (int32_t*)malloc(bytesA_local);     // Linhas locais de A
    int32_t *C_loc  = (int32_t*)malloc(bytesC_local);     // Linhas locais de C

    if (!B || !BT || (!A_loc && rows_local > 0) || (!C_loc && rows_local > 0)) {
        fprintf(stderr, "[%d] Falha malloc (B/BT/A_loc/C_loc)\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ===== GERAÇÃO DE DADOS (APENAS RANK 0) =====
    // Só o rank 0 gera ou carrega as matrizes completas
    if (rank == 0) {
        // Aloca matriz A completa
        A_root = (int32_t*)malloc((size_t)N * N * sizeof(int32_t));
        if (!A_root) {
            fprintf(stderr, "Falha malloc A_root.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Preenche A e B com números aleatórios
        srand(12345u);
        for (int i = 0; i < N * N; ++i) {
            A_root[i] = rand() % 2;  // Valores 0 ou 1
            B[i]      = rand() % 2;  // Valores 0 ou 1
        }
    }

    // ===== PREPARA ARRAYS PARA SCATTERV/GATHERV =====
    // MPI_Scatterv/Gatherv operam em elementos, não em linhas
    // Precisa converter: linhas -> elementos
    int *sendcounts_elems = (int*)malloc(size * sizeof(int));  // Elementos por processo
    int *displs_elems     = (int*)malloc(size * sizeof(int));  // Deslocamento em elementos
    if (!sendcounts_elems || !displs_elems) {
        if (rank == 0) fprintf(stderr, "Falha malloc sendcounts/displs elems\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Converte counts (em linhas) para counts em elementos
    // linhas -> elementos = linhas * N (N colunas por linha)
    for (int r = 0; r < size; ++r) {
        sendcounts_elems[r] = counts[r] * N;   // Elementos = linhas * colunas
        displs_elems[r]     = displs[r] * N;   // Deslocamento em elementos
    }

    // ===== SINCRONIZAÇÃO =====
    // Todos os ranks aguardam aqui para começar a medir tempo
    // Garante que todos começam juntos
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();  // Início da medição

    // ===== DISTRIBUIÇÃO: Scatterv =====
    // MPI_Scatterv: Rank 0 distribui as linhas de A para todos os ranks
    // sendcounts_elems[r] = quantos elementos rank r vai receber
    // displs_elems[r] = de onde tirar em A_root
    // A_loc = onde colocar neste rank
    MPI_Scatterv(A_root, sendcounts_elems, displs_elems, MPI_INT32_T,
                 A_loc, rows_local * N, MPI_INT32_T,
                 0, MPI_COMM_WORLD);

    // ===== BROADCAST DE B =====
    // MPI_Bcast: Rank 0 envia a matriz B completa para todos
    // Todos os ranks precisam de B completa para multiplicar
    MPI_Bcast(B, N * N, MPI_INT32_T, 0, MPI_COMM_WORLD);

    // ===== TRANSPOSIÇÃO LOCAL DE B =====
    // Cada rank transpoe B em sua memória local
    // Razão: BT[j][k] = B[k][j] permite acessos mais sequenciais na memória
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            BT[(size_t)j * N + i] = B[(size_t)i * N + j];
        }
    }

    // ===== INICIALIZAÇÃO DE C =====
    // C_loc começa com zeros
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows_local * N; ++i) {
        C_loc[i] = 0;
    }

    // ===== MULTIPLICAÇÃO EM BLOCOS (HYBRID MPI+OpenMP) =====
    // Cada rank calcula apenas suas linhas (rows_local)
    // Usa OpenMP para paralelizar os blocos dentro de cada rank
    // #pragma omp parallel for schedule(static): Distribui blocos entre threads
    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < rows_local; ii += T) {  // Blocos de linha
        for (int jj = 0; jj < N; jj += T) {       // Blocos de coluna
            for (int kk = 0; kk < N; kk += T) {   // Blocos de profundidade

                // Limites do bloco
                const int i_max = min_i(ii + T, rows_local);
                const int j_max = min_i(jj + T, N);
                const int k_max = min_i(kk + T, N);

                // Processa cada linha do bloco
                for (int i = ii; i < i_max; ++i) {
                    // restrict: dica ao compilador que não há aliasing
                    int32_t *restrict Ci =
                        &C_loc[(size_t)i * N + jj];
                    const int32_t *restrict Ai =
                        &A_loc[(size_t)i * N + kk];

                    // Processa cada coluna do bloco
                    for (int j = jj; j < j_max; ++j) {
                        const int32_t *restrict BTj =
                            &BT[(size_t)j * N + kk];
                        int sum = 0;

                        // SIMD: Vetorização do produto interno
                        #pragma omp simd reduction(+:sum)
                        for (int k = kk; k < k_max; ++k) {
                            // Ai[k-kk] = A_loc[i][k], BTj[k-kk] = BT[j][k]
                            sum += Ai[k - kk] * BTj[k - kk];
                        }
                        Ci[j - jj] += sum;
                    }
                }
            }
        }
    }

    // ===== REUNIÃO: Gatherv =====
    // Apenas rank 0 aloca espaço para C completa
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
