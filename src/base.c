#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define NUMERO 2000

int32_t* matriz1;
int32_t* matriz2;
int32_t* resultado;

int _seed = 42;
int _matrix_type = 0;
int _matrix_length = NUMERO;

// Função para gerar matrizes com valores aleatórios
void gerar_matrizes() {
    srand(_seed);

    for (int linha = 0; linha < _matrix_length; linha++) {
        for (int coluna = 0; coluna < _matrix_length; coluna++) {
            matriz1[linha * _matrix_length + coluna] = rand() % 2;
            matriz2[linha * _matrix_length + coluna] = rand() % 2;
        }
    }
}

int carregar_matriz(const char *arquivo, int32_t *matriz) {
    FILE *fp = fopen(arquivo, "r");
    if (!fp) {
        perror("Erro ao abrir arquivo");
        return 1;
    }

    for (int i = 0; i < _matrix_length; i++) {
        for (int j = 0; j < _matrix_length; j++) {
            if (fscanf(fp, "%d,", &matriz[i * _matrix_length + j]) != 1) {
                printf("Erro de leitura na pos (%d,%d)\n", i, j);
                fclose(fp);
                return 1;
            }
        }
    }

    fclose(fp);
    return 0;
}

// Função para multiplicar matrizes
void multiplicar_matrizes() {
    //clock_t ini, end;
    //ini = clock();
    
    for (int linha = 0; linha < _matrix_length; linha++) {
        for (int coluna = 0; coluna < _matrix_length; coluna++) {
            int32_t soma = 0;
            for (int k = 0; k < _matrix_length; k++) {
                soma += matriz1[linha * _matrix_length + k] * matriz2[k * _matrix_length + coluna];
            }
            resultado[linha * _matrix_length + coluna] = soma;
        }
    }
    //end = clock();
    //printf("%.2f s\n", (float) (end - ini) / CLOCKS_PER_SEC);
}

long long calculate_checksum()
{
    long long checksum = 0;
    for (int i = 0; i < _matrix_length; ++i) {
        for (int j = 0; j < _matrix_length; ++j) {
            checksum += (long long) resultado[i * _matrix_length + j];
        }
    }

    return checksum;
}

void get_args(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) 
    {
        if ((strcmp(argv[i], "--length") == 0 || strcmp(argv[i], "--l") == 0) && i + 1 < argc) {
            _matrix_length = atoi(argv[i + 1]);
            i++;
        }
        else if ((strcmp(argv[i], "--seed") == 0 
               || strcmp(argv[i], "--s") == 0) && i + 1 < argc) {
            _seed = atoi(argv[i + 1]);
            i++;
        }
        else if ((strcmp(argv[i], "--matrix_type") == 0 || strcmp(argv[i], "--mt") == 0) && i + 1 < argc) {
            _matrix_type = atoi(argv[i + 1]);
            i++;
        }
    }
}

int main(int argc, char **argv) {

    get_args(argc, argv);

    matriz1 = malloc(_matrix_length * _matrix_length * sizeof(int32_t));
    matriz2 = malloc(_matrix_length * _matrix_length * sizeof(int32_t));
    resultado = malloc(_matrix_length * _matrix_length * sizeof(int32_t));

    if (_matrix_type)
    {
        if (carregar_matriz("matriz1.csv", matriz1))
        {
            fprintf(stderr, "Erro ao carregar matriz1.csv\n");
            return 1;
        }

        if (carregar_matriz("matriz2.csv", matriz2))
        {
            fprintf(stderr, "Erro ao carregar matriz2.csv\n");
            return 1;
        }
    }
    else
    {
        gerar_matrizes(_matrix_length);
    }

    multiplicar_matrizes(_matrix_length);

    fprintf(stdout, "%lld", calculate_checksum(_matrix_length));

    return 0;
}