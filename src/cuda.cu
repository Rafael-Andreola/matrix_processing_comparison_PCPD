// ====== INCLUSÕES DE BIBLIOTECAS ======
#include <cstdio>        // Entrada/saída
#include <cstdlib>       // Funções gerais (malloc, rand)
#include <cstdint>       // Tipos inteiros com tamanho fixo (int32_t)
#include <cuda_runtime.h>  // API de runtime CUDA

// ====== CONSTANTES ======
// Tamanho padrão da matriz (4000 x 4000)
#ifndef NUMERO
#define NUMERO 4000
#endif

// Tamanho do bloco CUDA (threads por dimensão)
// Valores 16/32 funcionam bem; 32x32=1024 threads por bloco
#ifndef BLOCK
#define BLOCK 32
#endif

// ====== VARIÁVEIS GLOBAIS ======
int _matrix_type = 0;    // Tipo: 0 = gerar aleatório, 1 = carregar de arquivo
int _seed = 42;          // Seed para números aleatórios
int _matrix_length = NUMERO;  // Tamanho da matriz

// ====== KERNEL CUDA: Multiplicação de Matrizes com Tiling em Shared Memory ======
// DESCRIÇÃO: Executa multiplicação C = A * B em paralelo na GPU
// Estratégia: Tiling - divide A e B em blocos pequenos que cabem em shared memory
//            Isso reduz acessos à memória global da GPU (muito mais rápido)
//
// ENTRADA:
//   A: Matriz A (NxN) em memória global
//   B: Matriz B (NxN) em memória global
//   C: Matriz resultado (NxN) em memória global
//   N: Tamanho da matriz
//
// SAÍDA: C modificada com o resultado da multiplicação
// 
// EXECUÇÃO:
//   - blockIdx.x, blockIdx.y: Identifica qual bloco está sendo executado
//   - threadIdx.x, threadIdx.y: Id da thread dentro do bloco
//   - Cada thread calcula um elemento de C
__global__ void matmul_tiled(const int32_t *__restrict__ A,
                             const int32_t *__restrict__ B,
                             int32_t *__restrict__ C,
                             int N)
{
    // Alocação de memória compartilhada (shared memory)
    // Cada thread-block tem uma cópia desses blocos
    // __shared__: visível para todas as threads do mesmo bloco, rápido acesso
    __shared__ int32_t As[BLOCK][BLOCK];  // Bloco da matriz A
    __shared__ int32_t Bs[BLOCK][BLOCK];  // Bloco da matriz B

    // Coordenadas globais da thread (qual elemento de C ela calculará)
    const int row = blockIdx.y * BLOCK + threadIdx.y;  // Linha de C
    const int col = blockIdx.x * BLOCK + threadIdx.x;  // Coluna de C

    // Acumulador para armazenar a soma do produto
    int32_t acc = 0;

    // Número de tiles (blocos) ao longo da dimensão K
    // Se N=1024 e BLOCK=32, temos 32 tiles
    const int tiles = (N + BLOCK - 1) / BLOCK;

    // ===== LOOP PRINCIPAL: Itera sobre todos os tiles =====
    for (int t = 0; t < tiles; ++t)
    {
        // Coluna base do tile de A para a thread atual
        const int aCol = t * BLOCK + threadIdx.x;  // Qual coluna de A usar
        // Linha base do tile de B para a thread atual
        const int bRow = t * BLOCK + threadIdx.y;  // Qual linha de B usar

        // ===== CARREGAMENTO DO TILE DE A =====
        // Cada thread carrega um elemento: A[row][aCol]
        // __restrict__: compiler hint que A não sofre aliasing
        if (row < N && aCol < N)
            // Carrega de memória global para shared memory
            As[threadIdx.y][threadIdx.x] = A[row * N + aCol];
        else
            // Se fora dos limites, coloca 0 (não afeta resultado)
            As[threadIdx.y][threadIdx.x] = 0;

        // ===== CARREGAMENTO DO TILE DE B =====
        // Cada thread carrega um elemento: B[bRow][col]
        if (bRow < N && col < N)
            // Carrega de memória global para shared memory
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else
            // Se fora dos limites, coloca 0
            Bs[threadIdx.y][threadIdx.x] = 0;

        // ===== SINCRONIZAÇÃO =====
        // __syncthreads(): Aguarda todas as threads do bloco terminarem
        // Garante que todos os dados estão carregados antes de usar
        __syncthreads();

        // ===== MULTIPLICAÇÃO DO BLOCO =====
        // Multiplica os sub-blocos: soma(As[threadIdx.y][k] * Bs[k][threadIdx.x])
        // #pragma unroll: Pede ao compilador para desenrolar o loop (mais rápido)
#pragma unroll
        for (int k = 0; k < BLOCK; ++k)
        {
            // Acumula o produto de elementos do bloco
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        // Sincroniza novamente antes do próximo tile
        __syncthreads();
    }

    // ===== ARMAZENA O RESULTADO =====
    // Escreve o resultado acumulado de volta à memória global
    if (row < N && col < N)
        C[row * N + col] = acc;
}

// ====== FUNÇÃO: Processar Argumentos de Linha de Comando ======
// DESCRIÇÃO: Parseia argumentos passados ao programa
// Argumentos: --length, --seed, --matrix_type
// ENTRADA: argc, argv (argumentos de linha de comando)
// SAÍDA: Modifica variáveis globais
void get_args(int argc, char **argv)
{
    for (int i = 1; i < argc; i++)
    {
        // Processa --length ou --l
        if ((strcmp(argv[i], "--length") == 0 || strcmp(argv[i], "--l") == 0) && i + 1 < argc)
        {
            _matrix_length = atoi(argv[i + 1]);
            i++;
        }
        // Processa --seed ou --s
        else if ((strcmp(argv[i], "--seed") == 0 || strcmp(argv[i], "--s") == 0) && i + 1 < argc)
        {
            _seed = atoi(argv[i + 1]);
            i++;
        }
        // Processa --matrix_type ou --mt
        else if ((strcmp(argv[i], "--matrix_type") == 0 || strcmp(argv[i], "--mt") == 0) && i + 1 < argc)
        {
            _matrix_type = atoi(argv[i + 1]);
            i++;
        }
    }
}

// ====== FUNÇÃO: Verificar Erros CUDA ======
// DESCRIÇÃO: Função auxiliar para checar se houve erro em uma chamada CUDA
// Se houver erro, imprime mensagem e encerra o programa
// ENTRADA: code (código de retorno CUDA), file (arquivo de origem), line (linha)
// SAÍDA: Nenhuma (encerra se erro)
static inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        // Imprime erro com arquivo e linha onde ocorreu
        fprintf(stderr, "CUDA Error: %s (%s:%d)\n", cudaGetErrorString(code), file, line);
        exit(1);
    }
}
// Macro para simplicidade: CUDA_OK(cudaMalloc(...)) em vez de gpuAssert(...)
#define CUDA_OK(x) gpuAssert((x), __FILE__, __LINE__)

// ====== FUNÇÃO: Main (Ponto de Entrada) ======
// DESCRIÇÃO: Coordena todo o processo:
//   1. Parseia argumentos
//   2. Aloca memória no host (CPU) e device (GPU)
//   3. Gera dados ou carrega de arquivo
//   4. Executa kernel CUDA
//   5. Mede tempo com eventos CUDA
//   6. Cópia resultado de volta pro host
//   7. Calcula checksum e libera memória
int main(int argc, char **argv)
{
    // Parseia argumentos da linha de comando
    get_args(argc, argv);
    int N = _matrix_length;

    // Tamanho total em bytes de cada matriz (N x N elementos de int32_t)
    const size_t bytes = (size_t)N * (size_t)N * sizeof(int32_t);

    // ===== ALOCAÇÃO EM HOST (CPU) =====
    // Aloca espaço para as matrizes A e B (entrada) e C (resultado)
    int32_t *hA = (int32_t *)malloc(bytes);  // Matriz A no host
    int32_t *hB = (int32_t *)malloc(bytes);  // Matriz B no host
    int32_t *hC = (int32_t *)malloc(bytes);  // Matriz C no host (resultado)
    if (!hA || !hB || !hC)
    {
        fprintf(stderr, "Falha malloc host\n");
        return 1;
    }

    // ===== GERAÇÃO DE DADOS NO HOST =====
    srand(_seed);

    // Preenche as matrizes com números aleatórios (0-254)
    for (int i = 0; i < N * N; ++i)
    {
        hA[i] = rand() % 255;
        hB[i] = rand() % 255;
        hC[i] = 0;  // Inicializa C com zeros
    }

    // ===== ALOCAÇÃO NO DEVICE (GPU) =====
    // Aloca memória na GPU para as mesmas matrizes
    int32_t *dA, *dB, *dC;
    CUDA_OK(cudaMalloc(&dA, bytes));  // Matriz A na GPU
    CUDA_OK(cudaMalloc(&dB, bytes));  // Matriz B na GPU
    CUDA_OK(cudaMalloc(&dC, bytes));  // Matriz C na GPU

    // ===== CÓPIA DE DADOS DO HOST PARA DEVICE =====
    // Copia A e B do host para GPU
    CUDA_OK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
    // Inicializa C na GPU com zeros
    CUDA_OK(cudaMemset(dC, 0, bytes));

    // ===== CONFIGURAÇÃO DA GRADE DE EXECUÇÃO =====
    // block: Tamanho de cada thread-block (BLOCK x BLOCK = 32x32 = 1024 threads)
    dim3 block(BLOCK, BLOCK);
    // grid: Número de blocos (ceil(N/BLOCK) x ceil(N/BLOCK))
    // Para N=4000, BLOCK=32: grid = (125, 125) = 15625 blocos
    dim3 grid((N + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);

    // ===== MEDIÇÃO DE TEMPO COM EVENTOS CUDA =====
    // Eventos CUDA são checkpoints para medir tempo de execução
    cudaEvent_t e0, e1;  // Eventos: início e fim
    CUDA_OK(cudaEventCreate(&e0));
    CUDA_OK(cudaEventCreate(&e1));

    // ===== EXECUÇÃO DO KERNEL =====
    // Registra início
    CUDA_OK(cudaEventRecord(e0));
    // Lancia o kernel com grid e block
    // <<<grid, block>>> = (quantos blocos em 2D, quantas threads por bloco em 2D)
    matmul_tiled<<<grid, block>>>(dA, dB, dC, N);
    // Registra fim
    CUDA_OK(cudaEventRecord(e1));
    // Aguarda que o kernel termine
    CUDA_OK(cudaEventSynchronize(e1));
    // Verifica se houve erro durante execução
    CUDA_OK(cudaPeekAtLastError());

    // Calcula tempo decorrido entre e0 e e1
    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));  // Retorna tempo em milissegundos

    // ===== CÓPIA DO RESULTADO DO DEVICE PARA HOST =====
    // Copia matriz C de volta da GPU para CPU
    CUDA_OK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    // ===== CÁLCULO DO CHECKSUM =====
    // Soma todos os elementos para verificação de integridade
    long long checksum = 0;
    for (int i = 0; i < N * N; ++i)
        checksum += hC[i];

    // ===== SAÍDA =====
    // Imprime tempo de execução em segundos (ms / 1000.0)
    printf("%.5f s\n", ms / 1000.0f);
    // Imprime checksum no stderr para comparação
    fprintf(stderr, "checksum=%lld\n", checksum);

    // ===== LIBERAÇÃO DE MEMÓRIA =====
    // Libera memória na GPU
    CUDA_OK(cudaFree(dA));
    CUDA_OK(cudaFree(dB));
    CUDA_OK(cudaFree(dC));
    // Libera memória no host
    free(hA);
    free(hB);
    free(hC);
    // Destrói eventos CUDA
    CUDA_OK(cudaEventDestroy(e0));
    CUDA_OK(cudaEventDestroy(e1));
    
    return 0;
}
