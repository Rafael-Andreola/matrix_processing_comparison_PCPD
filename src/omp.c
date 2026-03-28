#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#ifndef NUMERO
#define NUMERO 4000
#endif

#ifndef TILE
#define TILE 128
#endif

int _matrix_type = 0;
int _seed = 42;
int _matrix_length = NUMERO;

static inline void *xaligned_alloc(size_t align, size_t bytes)
{
#if defined(_WIN32)
    return _aligned_malloc(bytes, align);
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    // aligned_alloc exige múltiplo de align
    size_t sz = (bytes + (align - 1)) & ~(align - 1);
    return aligned_alloc(align, sz);
#else
    void *p = NULL;
    if (posix_memalign(&p, align, bytes) != 0)
        return NULL;
    return p;
#endif

    return malloc(bytes);
}

static inline void xaligned_free(void *p)
{
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

static int carregar_matriz(const char *arquivo, int32_t *matriz)
{
    FILE *fp = fopen(arquivo, "r");
    if (!fp)
    {
        perror("Erro ao abrir arquivo");
        return 1;
    }

    for (int i = 0; i < _matrix_length; i++)
    {
        for (int j = 0; j < _matrix_length; j++)
        {
            if (fscanf(fp, "%d,", &matriz[i * _matrix_length + j]) != 1)
            {
                printf("Erro de leitura na pos (%d,%d)\n", i, j);
                fclose(fp);
                return 1;
            }
        }
    }

    fclose(fp);
    return 0;
    ;
}

static inline int min_i(int a, int b) { return a < b ? a : b; }

void generate_matrix(int32_t *A, int32_t *B)
{
    // Geração dos dados
    srand(_seed);

#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < _matrix_length; i++)
    {
        for (int j = 0; j < _matrix_length; j++)
        {
            A[(size_t)i * _matrix_length + j] = rand() % 255;
            B[(size_t)i * _matrix_length + j] = rand() % 255;
        }
    }
}

void calculate_matrix(int32_t *A, int32_t *B, int32_t *BT, int32_t *C)
{
    const double t0 = omp_get_wtime();
// Transpor B -> BT para acesso contíguo no loop interno
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < _matrix_length; i++)
        for (int j = 0; j < _matrix_length; j++)
            BT[(size_t)j * _matrix_length + i] = B[(size_t)i * _matrix_length + j];

// Zerar C
#pragma omp parallel for schedule(static)
    for (int i = 0; i < _matrix_length * _matrix_length; i++)
        C[i] = 0;

// Multiplicação bloqueada: C += A * B, usando BT
// Paralelismo nos blocos externos, vetoriza o loop de k
#pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < _matrix_length; ii += TILE)
    {
        for (int jj = 0; jj < _matrix_length; jj += TILE)
        {
            for (int kk = 0; kk < _matrix_length; kk += TILE)
            {
                const int i_max = min_i(ii + TILE, _matrix_length);
                const int j_max = min_i(jj + TILE, _matrix_length);
                const int k_max = min_i(kk + TILE, _matrix_length);
                for (int i = ii; i < i_max; i++)
                {
                    const int32_t *Ai = &A[(size_t)i * _matrix_length + kk];
                    int32_t *Ci = &C[(size_t)i * _matrix_length + jj];
                    for (int j = jj; j < j_max; j++)
                    {
                        const int32_t *BTj = &BT[(size_t)j * _matrix_length + kk];
                        int sum = 0;

#pragma omp simd reduction(+ : sum)
                        for (int k = kk; k < k_max; k++)
                        {
                            // Ai[k-kk] == A[i, k], BTj[k-kk] == B[k, j]
                            sum += Ai[k - kk] * BTj[k - kk];
                        }
                        Ci[j - jj] += sum;
                    }
                }
            }
        }
    }

    const double t1 = omp_get_wtime();
    printf("%.5f s\n", t1 - t0);
}

void calculate_checksum(int32_t *C)
{
    long long checksum = 0;
#pragma omp parallel for reduction(+ : checksum) schedule(static)
    for (int i = 0; i < _matrix_length * _matrix_length; i++)
        checksum += C[i];

    fprintf(stdout, "%lld", checksum);
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

int main(int argc, char **argv)
{
    get_args(argc, argv);

    const size_t bytes = (size_t)_matrix_length * (size_t)_matrix_length * sizeof(int32_t);

    // Alocação alinhada para melhor SIMD e pré-busca
    int32_t *A = (int32_t *)xaligned_alloc(64, bytes);
    int32_t *B = (int32_t *)xaligned_alloc(64, bytes);
    int32_t *BT = (int32_t *)xaligned_alloc(64, bytes); // B transposta
    int32_t *C = (int32_t *)xaligned_alloc(64, bytes);

    if (!A || !B || !BT || !C)
    {
        fprintf(stderr, "Falha ao alocar memoria\n");
        return 1;
    }

    if (_matrix_type)
    {
        if (carregar_matriz("matriz1.csv", A))
        {
            fprintf(stderr, "Erro ao carregar matriz1.csv\n");
            return 1;
        }

        if (carregar_matriz("matriz2.csv", B))
        {
            fprintf(stderr, "Erro ao carregar matriz2.csv\n");
            return 1;
        }
    }
    else
    {
        generate_matrix(A, B);
    }

    printf("%d, %d\n", A[0], A[1]);
    printf("%d, %d\n", B[0], B[1]);

    calculate_matrix(A, B, BT, C);

    // Evita que o compilador descarte C
    calculate_checksum(C);

    xaligned_free(A);
    xaligned_free(B);
    xaligned_free(BT);
    xaligned_free(C);
    return 0;
}
