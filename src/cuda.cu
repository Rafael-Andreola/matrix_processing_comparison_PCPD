#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#ifndef NUMERO
#define NUMERO 4000
#endif

#ifndef BLOCK
#define BLOCK 32 // 16/32 funcionam bem, ajustar conforme GPU
#endif

int _matrix_type = 0;
int _seed = 42;
int _matrix_length = NUMERO;

// Kernel: C = A * B  (N x N), int32, tiling em shared memory
__global__ void matmul_tiled(const int32_t *__restrict__ A,
                             const int32_t *__restrict__ B,
                             int32_t *__restrict__ C,
                             int N)
{
    __shared__ int32_t As[BLOCK][BLOCK];
    __shared__ int32_t Bs[BLOCK][BLOCK];

    const int row = blockIdx.y * BLOCK + threadIdx.y;
    const int col = blockIdx.x * BLOCK + threadIdx.x;

    int32_t acc = 0;

    // Número de tiles ao longo de K
    const int tiles = (N + BLOCK - 1) / BLOCK;

    for (int t = 0; t < tiles; ++t)
    {
        // Coluna base do tile de A e linha base do tile de B
        const int aCol = t * BLOCK + threadIdx.x;
        const int bRow = t * BLOCK + threadIdx.y;

        // Carrega tile de A: [row, aCol .. aCol+BLOCK)
        if (row < N && aCol < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0;

        // Carrega tile de B: [bRow .. bRow+BLOCK, col]
        if (bRow < N && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

// Multiplica os sub-blocos (loop no eixo K do tile)
#pragma unroll
        for (int k = 0; k < BLOCK; ++k)
        {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = acc;
}

void get_args(int argc, char **argv)
{
    for (int i = 1; i < argc; i++)
    {
        if ((strcmp(argv[i], "--length") == 0 || strcmp(argv[i], "--l") == 0) && i + 1 < argc)
        {
            _matrix_length = atoi(argv[i + 1]);
            i++;
        }
        else if ((strcmp(argv[i], "--seed") == 0 || strcmp(argv[i], "--s") == 0) && i + 1 < argc)
        {
            _seed = atoi(argv[i + 1]);
            i++;
        }
        else if ((strcmp(argv[i], "--matrix_type") == 0 || strcmp(argv[i], "--mt") == 0) && i + 1 < argc)
        {
            _matrix_type = atoi(argv[i + 1]);
            i++;
        }
    }
}

static inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s (%s:%d)\n", cudaGetErrorString(code), file, line);
        exit(1);
    }
}
#define CUDA_OK(x) gpuAssert((x), __FILE__, __LINE__)
int main(int argc, char **argv)
{

    get_args(argc, argv);
    int N = _matrix_length;

    const size_t bytes = (size_t)N * (size_t)N * sizeof(int32_t);

    // Host
    int32_t *hA = (int32_t *)malloc(bytes);
    int32_t *hB = (int32_t *)malloc(bytes);
    int32_t *hC = (int32_t *)malloc(bytes);
    if (!hA || !hB || !hC)
    {
        fprintf(stderr, "Falha malloc host\n");
        return 1;
    }

    srand(_seed);

    for (int i = 0; i < N * N; ++i)
    {
        hA[i] = rand() % 255;
        hB[i] = rand() % 255;
        hC[i] = 0;
    }

    // Device
    int32_t *dA, *dB, *dC;
    CUDA_OK(cudaMalloc(&dA, bytes));
    CUDA_OK(cudaMalloc(&dB, bytes));
    CUDA_OK(cudaMalloc(&dC, bytes));

    CUDA_OK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(dC, 0, bytes));

    // Grade de execução
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);

    // Timer
    cudaEvent_t e0, e1;
    CUDA_OK(cudaEventCreate(&e0));
    CUDA_OK(cudaEventCreate(&e1));

    CUDA_OK(cudaEventRecord(e0));
    matmul_tiled<<<grid, block>>>(dA, dB, dC, N);
    CUDA_OK(cudaEventRecord(e1));
    CUDA_OK(cudaEventSynchronize(e1));
    CUDA_OK(cudaPeekAtLastError());

    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, e0, e1)); // milissegundos

    CUDA_OK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    // Checksum para evitar dead-code
    long long checksum = 0;
    for (int i = 0; i < N * N; ++i)
        checksum += hC[i];

    printf("%.5f s\n", ms / 1000.0f);
    fprintf(stderr, "checksum=%lld\n", checksum);

    // Libera
    CUDA_OK(cudaFree(dA));
    CUDA_OK(cudaFree(dB));
    CUDA_OK(cudaFree(dC));
    free(hA);
    free(hB);
    free(hC);
    CUDA_OK(cudaEventDestroy(e0));
    CUDA_OK(cudaEventDestroy(e1));
    return 0;
}
