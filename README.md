// ...existing code...

# trabalho-processamento-matrizes

Repositório com implementações de multiplicação de matrizes (base, OpenMP, MPI e CUDA),
ferramentas de comparação e dados de teste.

## Arquivos principais:
- [src/base.c](src/base.c) — implementação sequencial (funções-chave: [`gerar_matrizes`](src/base.c),       [`multiplicar_matrizes`](src/base.c), [`calculate_checksum`](src/base.c), [`main`](src/base.c))
- [src/omp.c](src/omp.c) — implementação com OpenMP (funções-chave: [`generate_matrix`](src/omp.c), [`calculate_matrix`](src/omp.c), [`calculate_checksum`](src/omp.c))
- [src/mpi.c](src/mpi.c) — implementação distribuída com MPI (ex.: distribuição de linhas, gather/scatter)
- [src/cuda.cu](src/cuda.cu) — implementação CUDA (kernel: [`matmul_tiled`](src/cuda.cu))
- [tests/compare.py](tests/compare.py) — script Python para gerar matrizes idênticas e comparar duas execuções
- [LICENSE](LICENSE) — licença MIT
- [.gitignore](.gitignore) — arquivos ignorados (executáveis, CSVs, IDE)

## Requisitos
- compilador C (gcc/clang) com suporte a OpenMP
- mpicc/mpiexec para MPI
- NVCC + drivers CUDA (se usar CUDA)
- Python 3 + numpy (para testes)
- Windows: quando compilar CUDA use "x64 Native Tools Command Prompt for VS"

## Como compilar
- Base (sequencial)
  - gcc -O3 src/base.c -o base.exe
- OpenMP
  - gcc -O3 -fopenmp -march=native src/omp.c -o omp.exe
- MPI
  - mpicc -O3 -Ofast -fopenmp -march=native -funroll-loops src/mpi.c -o mpi.exe
- CUDA
  - nvcc -O3 -arch=native src/cuda.cu -o cuda.exe

## Como executar (exemplos)
- Flags comuns:
  - --l / --length N  → tamanho N x N
  - --s / --seed S    → seed aleatória
  - --mt / --matrix_type MT → 0 = gera aleatoriamente a partir da seed; 1 = carrega arquivos matriz1.csv e matriz2.csv
- Executável base:
  - ./base.exe --l 2000 --s 45
- OpenMP:
  - ./omp.exe --l 2000 --s 40
- MPI (exemplo):
  - mpiexec -hostfile hosts -np 8 --bind-to core ./mpi.exe --length 4000 --tile 128
- CUDA:
  - ./cuda.exe --l 4000
  - O binário imprime tempo wall/CPU e um checksum.

## Testes e comparação
- O script [tests/compare.py](tests/compare.py) gera duas cópias idênticas de matriz em CSV e executa dois binários para comparar a saída:
  - python tests/compare.py --exe1 base.exe --exe2 omp.exe --length 2000
- Observação: [tests/compare.py](tests/compare.py) espera encontrar os executáveis passados e salva arquivos `matriz1.csv`/`matriz2.csv` (esses arquivos são ignorados pelo [.gitignore](.gitignore)).

## Saída e verificação
- As implementações normalmente imprimem:
  - tempo de execução (stdout ou stderr dependendo do código)
  - checksum do resultado (usado para garantir correção)
- Para depuração, verifique stderr caso a execução falhe.

## Boas práticas / dicas
- Ajuste parâmetros de tile/bloco (TILE/BLOCK) conforme hardware.
- Em máquinas com muitas threads, experimente OMP_NUM_THREADS.
- Para comparar corretude entre implementações use [tests/compare.py](tests/compare.py) para garantir mesmas matrizes.
- Informações experimentais e tempos estão no próprio README original (historico em arquivos).