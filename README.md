# Comparação de Consumo de Energia: CPU vs GPU em Computadores Domésticos

Repositório com múltiplas implementações de multiplicação de matrizes para comparar desempenho e consumo de energia entre diferentes paradigmas de computação.

## 📋 Descrição do Projeto

Este projeto implementa multiplicação de matrizes de forma otimizada em:
- **Sequencial**: implementação base para referência
- **OpenMP**: paralelismo em memória compartilhada (CPU multi-core)
- **CUDA**: processamento em GPU

Todas as implementações produzem resultados idênticos (verificados por checksum) para permitir comparação justa de tempo de execução e eficiência energética.

## 📁 Estrutura do Projeto

```
src/
├── base.c      → Implementação sequencial (referência)
├── omp.c       → Implementação com OpenMP
└── cuda.cu     → Implementação CUDA (GPU)

tests/
└── compare.py  → Script para comparar duas implementações

LICENSE         → Licença MIT
README.md       → Este arquivo
```

## 🛠️ Pré-requisitos

### Obrigatório (Todos os programas)
- **Compilador C** (GCC ou Clang) com C11+
- **Python 3** (para testes)
- **NumPy** (para testes): `pip install numpy`

### Para OpenMP
- GCC/Clang com suporte a OpenMP (geralmente incluído)
- Flag: `-fopenmp`

### Para CUDA (GPU)
- **NVIDIA CUDA Toolkit** (versão 11.0+)
- **Drivers NVIDIA** atualizados
- **Visual Studio 2019+** (se no Windows)
- **Windows**: usar **"x64 Native Tools Command Prompt for VS"** para compilar CUDA

## 🔨 Como Compilar

### 1️⃣ Versão Sequencial (Base)
```bash
gcc -O3 src/base.c -o base.exe
```

### 2️⃣ Versão OpenMP
```bash
gcc -O3 -fopenmp -march=native src/omp.c -o omp.exe
```

### 3️⃣ Versão CUDA
```bash
# Windows (usar x64 Native Tools Command Prompt)
nvcc -O3 -arch=sm_75 src/cuda.cu -o cuda.exe

# Linux/macOS (adapte -arch conforme sua GPU)
nvcc -O3 -arch=native src/cuda.cu -o cuda.exe
```
---

## 🚀 Como Executar

### Opções Comuns para Todos os Programas
```
--l, --length N        → Tamanho da matriz (N × N)
--s, --seed S          → Seed para gerar números aleatórios (padrão: aleatório)
--mt, --matrix_type MT → 0 = gera aleatoriamente; 1 = carrega de arquivo CSV
```

### Exemplo: Versão Base
```bash
./base.exe --l 2000 --seed 45
```

### Exemplo: Versão OpenMP
```bash
./omp.exe --l 2000 --seed 40
```

### Exemplo: Versão CUDA
```bash
./cuda.exe --l 4000 --seed 42
```

---

## ✅ O que Esperar da Saída

Cada programa imprime:
- **Tempo de execução** (wall-clock ou CPU time)
- **Checksum** do resultado (para verificar correção)

---

## 📈 Resultados dos Testes de Energia

Os resultados dos testes de consumo de energia podem ser encontrados em [tests/energy_tests_results.zip](tests/energy_tests_results.zip). Este arquivo contém dados detalhados sobre o consumo energético de cada implementação (sequencial, OpenMP e CUDA) com diferentes tamanhos de matriz.

---

## 🧪 Testando e Comparando Implementações

O script [tests/compare.py](tests/compare.py) executa duas versões com **exatamente a mesma matriz** e compara os resultados.

### Compilar dois programas
```bash
gcc -O3 src/base.c -o base.exe
gcc -O3 -fopenmp -march=native src/omp.c -o omp.exe
```

### Executar comparação
```bash
python tests/compare.py --exe1 base.exe --exe2 omp.exe --length 2000
```

**O que faz:** Gera uma matriz 2000×2000, executa ambos os programas com a mesma matriz e compara os checksums.

---

## ⚡ Dicas e Otimizações

| Dica | Descrição |
|------|-----------|
| **Usar `-O3`** | Ativa todas as otimizações do compilador |
| **`-march=native`** | Usa instruções específicas de seu CPU (SSE, AVX, AVX2) |
| **`OMP_NUM_THREADS`** | Controla núcleos usados no OpenMP: `export OMP_NUM_THREADS=8` |
| **Testar diferentes tamanhos** | Comece com `--l 1000`, depois `2000`, `4000` |
| **Verificar GPU** | Use `nvidia-smi` para confirmar se CUDA está detectando GPU |

---

## 🔍 Troubleshooting

| Problema | Solução |
|----------|---------|
| CUDA não encontrado | Instale NVIDIA CUDA Toolkit ou adicione ao PATH |
| Resultados diferem entre versões | Verifique se está usando a mesma `--seed` em ambas |
| Checksum não bate | Possível overflow ou bug (abra issue no repositório), tente ao invés de utilizar a mesma seed, gerar um Excel com os mesmos valores e processar ele. |
| Muito lento (OpenMP) | CPU pode estar limitada; teste com matriz menor |

---

## 📊 Exemplo de Workflow Completo

```bash
# 1. Compilar tudo
gcc -O3 src/base.c -o base.exe
gcc -O3 -fopenmp -march=native src/omp.c -o omp.exe
nvcc -O3 -arch=sm_75 src/cuda.cu -o cuda.exe

# 2. Testar cada um (mesma matriz para comparação justa)
./base.exe --l 2000 --seed 123
./omp.exe --l 2000 --seed 123
./cuda.exe --l 2000 --seed 123

# 3. Comparar dois programas
python tests/compare.py --exe1 base.exe --exe2 omp.exe --length 2000

# 4. Testar com matrizes maiores (GPU fica mais vantajosa)
./base.exe --l 5000 --seed 456
./cuda.exe --l 5000 --seed 456
```

---

## 📝 Licença

MIT License - veja [LICENSE](LICENSE) para detalhes.
