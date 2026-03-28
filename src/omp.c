// ====== INCLUSÕES DE BIBLIOTECAS ======
#include <malloc.h>     // Alocação de memória alinhada
#include <stdio.h>      // Entrada/saída padrão
#include <stdlib.h>     // Funções gerais (malloc, rand, atoi)
#include <stdint.h>     // Tipos inteiros com tamanho fixo (int32_t)
#include <string.h>     // Funções de string (strcmp)
#include <time.h>       // Medição de tempo
#include <omp.h>        // OpenMP para paralelismo em memória compartilhada

// ====== CONSTANTES ======
// Tamanho padrão da matriz (4000 x 4000, maior que base.c para aproveitar paralelismo)
#ifndef NUMERO
#define NUMERO 4000
#endif

// Tamanho do tile/bloco para otimização de cache (128 elementos)
#ifndef TILE
#define TILE 128
#endif

// ====== VARIÁVEIS GLOBAIS (CONFIGURAÇÃO) ======
int _matrix_type = 0;    // Tipo: 0 = gerar aleatório, 1 = carregar de arquivo
int _seed = 42;          // Seed para números aleatórios
int _matrix_length = NUMERO;  // Tamanho da matriz (pode ser alterado por argumentos)

// ====== FUNÇÃO: Alocação de Memória Alinhada ======
// DESCRIÇÃO: Aloca memória alinhada para melhor desempenho de cache e SIMD
// Alinhamento a 64 bytes permite uso de instruções AVX-512
// ENTRADA: align (alinhamento em bytes), bytes (tamanho em bytes)
// SAÍDA: Ponteiro para memória alinhada
static inline void *xaligned_alloc(size_t align, size_t bytes)
{
#if defined(_WIN32)
    // Windows: usa _aligned_malloc
    return _aligned_malloc(bytes, align);
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    // C11: aligned_alloc exige que bytes seja múltiplo de align
    size_t sz = (bytes + (align - 1)) & ~(align - 1);
    return aligned_alloc(align, sz);
#else
    // Unix/Linux: usa posix_memalign
    void *p = NULL;
    if (posix_memalign(&p, align, bytes) != 0)
        return NULL;
    return p;
#endif

    // Fallback (raramente alcançado)
    return malloc(bytes);
}

// ====== FUNÇÃO: Libera Memória Alinhada ======
// DESCRIÇÃO: Libera memória alocada com xaligned_alloc
// ENTRADA: p (ponteiro para memória a liberar)
// SAÍDA: Nenhuma
static inline void xaligned_free(void *p)
{
#if defined(_WIN32)
    // Windows: usa _aligned_free
    _aligned_free(p);
#else
    // Unix/Linux: usa free padrão
    free(p);
#endif
}

// ====== FUNÇÃO: Carregar Matriz de Arquivo CSV ======
// DESCRIÇÃO: Lê uma matriz de um arquivo CSV (valores separados por vírgula)
// ENTRADA: arquivo (caminho do arquivo), matriz (ponteiro para armazenar dados)
// SAÍDA: 0 se sucesso, 1 se erro
static int carregar_matriz(const char *arquivo, int32_t *matriz)
{
    // Abre arquivo em modo leitura
    FILE *fp = fopen(arquivo, "r");
    if (!fp)
    {
        perror("Erro ao abrir arquivo");
        return 1;
    }

    // Loop duplo: lê cada elemento
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
}

// ====== FUNÇÃO AUX: Mínimo de Dois Inteiros ======
// DESCRIÇÃO: Função auxiliar inline para calcular min(a, b)
// ENTRADA: a, b (dois números inteiros)
// SAÍDA: O menor dos dois
static inline int min_i(int a, int b) { return a < b ? a : b; }

// ====== FUNÇÃO: Gerar Matrizes Aleatórias com OpenMP ======
// DESCRIÇÃO: Preenche duas matrizes A e B com valores aleatórios (0-254)
// Usa paralelismo OpenMP com collapse(2) para iterar 2D em paralelo
// ENTRADA: A, B (ponteiros para matrizes)
// SAÍDA: Nenhuma (modifica as matrizes A e B)
void generate_matrix(int32_t *A, int32_t *B)
{
    // Inicializa gerador de números aleatórios
    srand(_seed);

    // #pragma omp parallel for collapse(2) = Paralela o loop duplo
    // collapse(2) = combina os dois níveis de loop em um para melhor balanço de carga
    // schedule(static) = distribui iterações em blocos estáticos entre threads
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < _matrix_length; i++)
    {
        for (int j = 0; j < _matrix_length; j++)
        {
            // Gera valores aleatórios entre 0 e 254
            A[(size_t)i * _matrix_length + j] = rand() % 255;
            B[(size_t)i * _matrix_length + j] = rand() % 255;
        }
    }
}

// ====== FUNÇÃO: Multiplicar Matrizes com OpenMP ======
// DESCRIÇÃO: C = A * B com otimizações:
//   1. Transpor B para acesso linear na memória (cache-friendly)
//   2. Tiling/Bloqueamento para melhor localção de dados
//   3. Vetorização SIMD no loop mais interno
// Complexidade: O(n³) com otimizações de cache
// ENTRADA: A, B (matrizes de entrada), BT (B transposta, saída)
// SAÍDA: C (resultado da multiplicação)
void calculate_matrix(int32_t *A, int32_t *B, int32_t *BT, int32_t *C)
{
    // Inicia medição de tempo com OpenMP
    const double t0 = omp_get_wtime();
    
    // ===== ETAPA 1: Transpor B -> BT =====
    // Motivo: Se C[i][j] = sum(A[i][k] * B[k][j]), transpor B permite acessar
    //         BT[j][k] = B[k][j] de forma mais sequencial na memória
    // Paralelo: cada thread coloca B[i][j] em BT[j][i]
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < _matrix_length; i++)
        for (int j = 0; j < _matrix_length; j++)
            // BT[j][i] recebe o valor de B[i][j] (transposição)
            BT[(size_t)j * _matrix_length + i] = B[(size_t)i * _matrix_length + j];

    // ===== ETAPA 2: Zerar C =====
    // Inicializa C com zeros antes da soma acumulativa
#pragma omp parallel for schedule(static)
    for (int i = 0; i < _matrix_length * _matrix_length; i++)
        C[i] = 0;

    // ===== ETAPA 3: Multiplicação em Blocos com Tiling =====
    // Estratégia: Dividir a matriz em blocos de tamanho TILE x TILE
    // Vantagem: Blocos pequenos cabem em cache L2/L3, reduzindo miss de cache
    // Paralelismo: collapse(2) paralela os loops ii e jj (loops externos)
    // Vetorização: #pragma omp simd no loop k aplica instruções AVX/SSE
#pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < _matrix_length; ii += TILE)      // Blocos de linha
    {
        for (int jj = 0; jj < _matrix_length; jj += TILE)  // Blocos de coluna
        {
            for (int kk = 0; kk < _matrix_length; kk += TILE)  // Blocos de profundidade
            {
                // Define limites do bloco (trata matrizes que não são múltiplo de TILE)
                const int i_max = min_i(ii + TILE, _matrix_length);
                const int j_max = min_i(jj + TILE, _matrix_length);
                const int k_max = min_i(kk + TILE, _matrix_length);
                
                // Loop sobre linhas do bloco
                for (int i = ii; i < i_max; i++)
                {
                    // Ponteiro para linha i de A, começando em coluna kk
                    const int32_t *Ai = &A[(size_t)i * _matrix_length + kk];
                    // Ponteiro para linha i de C, começando em coluna jj
                    int32_t *Ci = &C[(size_t)i * _matrix_length + jj];
                    
                    // Loop sobre colunas do bloco
                    for (int j = jj; j < j_max; j++)
                    {
                        // Ponteiro para linha j de BT (que é coluna j de B), começando em kk
                        const int32_t *BTj = &BT[(size_t)j * _matrix_length + kk];
                        // Acumulador para C[i][j]
                        int sum = 0;

                        // SIMD: Vectorização do loop k com redução
                        // reduction(+ : sum) acumula resultados parciais de threads
#pragma omp simd reduction(+ : sum)
                        for (int k = kk; k < k_max; k++)
                        {
                            // Ai[k-kk] = A[i][k], BTj[k-kk] = B[k][j]=BT[j][k]
                            sum += Ai[k - kk] * BTj[k - kk];
                        }
                        // Acumula no resultado (pode vir de outros blocos de kk)
                        Ci[j - jj] += sum;
                    }
                }
            }
        }
    }

    // Final da medição de tempo
    const double t1 = omp_get_wtime();
    printf("%.5f s\n", t1 - t0);  // Imprime tempo em segundos
}

// ====== FUNÇÃO: Calcular Checksum da Matriz Resultado ======
// DESCRIÇÃO: Soma todos os elementos para verificação de integridade
// Usa OpenMP para paralelizar a soma com reduction
// ENTRADA: C (matriz resultado)
// SAÍDA: Imprime checksum em stdout
void calculate_checksum(int32_t *C)
{
    // Acumulador para a soma
    long long checksum = 0;
    
    // #pragma omp parallel for: Paralela o loop
    // reduction(+ : checksum): Cada thread tem sua cópia de checksum local,
    //                          ao final, soma todas as cópias
#pragma omp parallel for reduction(+ : checksum) schedule(static)
    for (int i = 0; i < _matrix_length * _matrix_length; i++)
        // Acumula cada elemento
        checksum += C[i];

    // Imprime o checksum final
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
