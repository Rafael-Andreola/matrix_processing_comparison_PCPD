// ====== INCLUSÕES DE BIBLIOTECAS ======
#include <stdio.h>      // Para entrada/saída (printf, fprintf, fscanf)
#include <stdlib.h>     // Para funções gerais (malloc, free, atoi, rand)
#include <stdint.h>     // Para tipos inteiros com tamanho fixo (int32_t)
#include <string.h>     // Para funções de string (strcmp)

// ====== CONSTANTES ======
#define NUMERO 2000     // Tamanho padrão da matriz (2000 x 2000)

// ====== VARIÁVEIS GLOBAIS (Matrizes) ======
int32_t* matriz1;       // Primeira matriz de entrada (alocada dinamicamente)
int32_t* matriz2;       // Segunda matriz de entrada (alocada dinamicamente)
int32_t* resultado;     // Matriz resultado da multiplicação (alocada dinamicamente)

// ====== VARIÁVEIS GLOBAIS (Configuração) ======
int _seed = 42;         // Seed para gerador de números aleatórios (padrão: 42)
int _matrix_type = 0;   // Tipo de matriz: 0 = gerar aleatória, 1 = carregar de arquivo
int _matrix_length = NUMERO;  // Tamanho da matriz (N x N), pode ser alterado por argumentos

// ====== FUNÇÃO: Gerar Matrizes Aleatórias ======
// DESCRIÇÃO: Preenche as matrizes 1 e 2 com valores aleatórios (0 ou 1)
// ENTRADA: Nenhuma (usa variáveis globais _seed e _matrix_length)
// SAÍDA: Nenhuma (modifica as matrizes globais matriz1 e matriz2)
void gerar_matrizes() {
    // Inicializa o gerador de números aleatórios com a seed fornecida
    srand(_seed);

    // Loop duplo para preencher cada elemento da matriz
    for (int linha = 0; linha < _matrix_length; linha++) {
        for (int coluna = 0; coluna < _matrix_length; coluna++) {
            // Calcula o índice linear no array 1D (row-major order)
            // Fórmula: índice = linha * largura + coluna
            // Gera número aleatório entre 0 e 1 (usando % 2)
            matriz1[linha * _matrix_length + coluna] = rand() % 2;
            matriz2[linha * _matrix_length + coluna] = rand() % 2;
        }
    }
}

// ====== FUNÇÃO: Carregar Matriz de Arquivo CSV ======
// DESCRIÇÃO: Lê uma matriz de um arquivo CSV (valores separados por vírgula)
// ENTRADA: 
//   - arquivo: caminho do arquivo CSV a ser lido
//   - matriz: ponteiro para o array onde será armazenada a matriz
// SAÍDA: 0 se sucesso, 1 se erro
int carregar_matriz(const char *arquivo, int32_t *matriz) {
    // Tenta abrir o arquivo em modo leitura
    FILE *fp = fopen(arquivo, "r");
    if (!fp) {
        // Se falhar, exibe mensagem de erro e retorna 1
        perror("Erro ao abrir arquivo");
        return 1;
    }

    // Loop duplo para ler cada elemento da matriz
    for (int i = 0; i < _matrix_length; i++) {
        for (int j = 0; j < _matrix_length; j++) {
            // fscanf lê um inteiro seguido de uma vírgula
            // Verifica se conseguiu ler exatamente 1 valor
            if (fscanf(fp, "%d,", &matriz[i * _matrix_length + j]) != 1) {
                printf("Erro de leitura na pos (%d,%d)\n", i, j);
                fclose(fp);  // Fecha o arquivo antes de sair
                return 1;    // Retorna erro
            }
        }
    }

    // Fecha o arquivo após leitura bem-sucedida
    fclose(fp);
    return 0;  // Retorna sucesso
}

// ====== FUNÇÃO: Multiplicar Matrizes ======
// DESCRIÇÃO: Realiza multiplicação de matriz1 por matriz2 e armazena em resultado
// Implementação: O(n³) - sequencial, sem otimizações de cache
// ENTRADA: Nenhuma (usa variáveis globais matriz1, matriz2, _matrix_length)
// SAÍDA: Nenhuma (modifica a matrix global resultado)
void multiplicar_matrizes() {
    // Para cada linha da matriz resultado
    for (int linha = 0; linha < _matrix_length; linha++) {
        // Para cada coluna da matriz resultado
        for (int coluna = 0; coluna < _matrix_length; coluna++) {
            // Inicializa acumulador para a soma do produto
            int32_t soma = 0;
            
            // Loop de multiplicação: soma(matriz1[linha][k] * matriz2[k][coluna])
            for (int k = 0; k < _matrix_length; k++) {
                // Acumula o produto dos elementos
                soma += matriz1[linha * _matrix_length + k] * matriz2[k * _matrix_length + coluna];
            }
            
            // Armazena o resultado na posição [linha][coluna]
            resultado[linha * _matrix_length + coluna] = soma;
        }
    }
    
    // NOTAS: Código comentado era para medir tempo de execução
    // clock_t ini, end;
    // ini = clock();
    // ... (código de multiplicação aqui) ...
    // end = clock();
    // printf("%.2f s\n", (float) (end - ini) / CLOCKS_PER_SEC);
}

// ====== FUNÇÃO: Calcular Checksum da Matriz Resultado ======
// DESCRIÇÃO: Soma todos os elementos da matriz resultado para verificação
// ENTRADA: Nenhuma (usa variáveis globais resultado, _matrix_length)
// SAÍDA: long long - soma de todos os elementos (usado para validação)
long long calculate_checksum()
{
    // Inicializa acumulador para armazenar a soma
    long long checksum = 0;
    
    // Loop duplo para iterar todos os elementos da matriz
    for (int i = 0; i < _matrix_length; ++i) {
        for (int j = 0; j < _matrix_length; ++j) {
            // Converte para long long para evitar overflow com int32_t
            // Soma cada elemento ao checksum
            checksum += (long long) resultado[i * _matrix_length + j];
        }
    }

    // Retorna a soma total (verificação de integridade)
    return checksum;
}

// ====== FUNÇÃO: Processar Argumentos de Linha de Comando ======
// DESCRIÇÃO: Faz parsing dos argumentos fornecidos ao executar o programa
// Argumentos suportados:
//   --length ou --l <N>       : Define tamanho da matriz (N x N)
//   --seed ou --s <S>         : Define seed para números aleatórios
//   --matrix_type ou --mt <T> : 0 = gerar aleatório, 1 = carregar arquivo
// ENTRADA: argc (número de argumentos), argv (vetor de strings com argumentos)
// SAÍDA: Nenhuma (modifica variáveis globais _matrix_length, _seed, _matrix_type)
void get_args(int argc, char **argv)
{
    // Loop através de todos os argumentos (começando em 1, pulando o nome do programa)
    for (int i = 1; i < argc; i++) 
    {
        // Verifica se é argumento de comprimento (--length ou --l)
        if ((strcmp(argv[i], "--length") == 0 || strcmp(argv[i], "--l") == 0) && i + 1 < argc) {
            // Converte o próximo argumento para inteiro e armazena
            _matrix_length = atoi(argv[i + 1]);
            i++;  // Pula para o próximo argumento
        }
        // Verifica se é argumento de seed (--seed ou --s)
        else if ((strcmp(argv[i], "--seed") == 0 
               || strcmp(argv[i], "--s") == 0) && i + 1 < argc) {
            // Converte o próximo argumento para inteiro e armazena
            _seed = atoi(argv[i + 1]);
            i++;  // Pula para o próximo argumento
        }
        // Verifica se é argumento de tipo de matriz (--matrix_type ou --mt)
        else if ((strcmp(argv[i], "--matrix_type") == 0 || strcmp(argv[i], "--mt") == 0) && i + 1 < argc) {
            // Converte o próximo argumento para inteiro e armazena
            _matrix_type = atoi(argv[i + 1]);
            i++;  // Pula para o próximo argumento
        }
    }
}

// ====== FUNÇÃO: Main (Ponto de Entrada do Programa) ======
// DESCRIÇÃO: Controla o fluxo principal: parse argumentos → aloca memória → 
//            carrega/gera matrizes → multiplica → calcula checksum → saída
// ENTRADA: argc (número de argumentos), argv (argumentos de linha de comando)
// SAÍDA: 0 se sucesso, 1 se erro
int main(int argc, char **argv) {

    // 1. Processa argumentos de linha de comando (altera variáveis globais)
    get_args(argc, argv);

    // 2. ALOCAÇÃO DE MEMÓRIA
    // Aloca espaço para matriz1 (N x N elementos de int32_t)
    matriz1 = malloc(_matrix_length * _matrix_length * sizeof(int32_t));
    
    // Aloca espaço para matriz2 (N x N elementos de int32_t)
    matriz2 = malloc(_matrix_length * _matrix_length * sizeof(int32_t));
    
    // Aloca espaço para matriz resultado (N x N elementos de int32_t)
    resultado = malloc(_matrix_length * _matrix_length * sizeof(int32_t));

    // 3. CARREGAMENTO OU GERAÇÃO DE DADOS
    // Se _matrix_type == 1, carrega de arquivos CSV
    if (_matrix_type)
    {
        // Tenta carregar matriz1 de arquivo
        if (carregar_matriz("matriz1.csv", matriz1))
        {
            fprintf(stderr, "Erro ao carregar matriz1.csv\n");
            return 1;  // Retorna erro se não conseguir carregar
        }

        // Tenta carregar matriz2 de arquivo
        if (carregar_matriz("matriz2.csv", matriz2))
        {
            fprintf(stderr, "Erro ao carregar matriz2.csv\n");
            return 1;  // Retorna erro se não conseguir carregar
        }
    }
    // Caso contrário, gera matrizes aleatórias
    else
    {
        gerar_matrizes();  // Preenche as matrizes com valores aleatórios
    }

    // 4. EXECUÇÃO DA MULTIPLICAÇÃO
    // Realiza a multiplicação de matriz1 por matriz2
    multiplicar_matrizes();

    // 5. CÁLCULO E SAÍDA DO CHECKSUM
    // Calcula a soma de todos os elementos do resultado (para validação)
    // e imprime diretamente na saída padrão
    fprintf(stdout, "%lld", calculate_checksum());

    // 6. RETORNO DO PROGRAMA
    return 0;  // Indica sucesso
}