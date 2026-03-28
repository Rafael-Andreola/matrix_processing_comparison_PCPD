import subprocess
import os
import sys
import time
import argparse
import numpy as np

def run_exe(exe_path, tamanho_matriz, seed):
    if not os.path.exists(exe_path):
        print(f"Erro: arquivo {exe_path} não existe.")
        return None

    print(f"Executando {exe_path}...")

    inicio = time.time()
    run = subprocess.run(
        [exe_path ,  "--l", str(tamanho_matriz), "--mt", "1", "--seed", str(seed)],
        capture_output=True,
        text=True
    )
    fim = time.time()

    if run.returncode != 0:
        print(f"Erro ao executar {exe_path}:")
        print(run.stderr)
        return None

    return (run.stdout.strip(), fim - inicio)


def main():
    parser = argparse.ArgumentParser(
        description="Compara a execução de dois arquivos .exe com a mesma matriz."
    )

    parser.add_argument("--exe1", required=True, help="Primeiro executável")
    parser.add_argument("--exe2", required=True, help="Segundo executável")
    parser.add_argument("--length", type=int, required=True, help="Tamanho da matriz NxN")
    #parser.add_argument(["--s", "--seed"], type=int, required=True, help="Seed para gerar matriz")

    args = parser.parse_args()

    exe1 = args.exe1
    exe2 = args.exe2
    tamanho_matriz = args.length if args.length is not None else args.l
    seed = 40

    print(f"Comparando {exe1} e {exe2} com tamanho de matriz {tamanho_matriz}.")

    m = np.random.randint(0, 2, size=(int(tamanho_matriz), int(tamanho_matriz)), dtype=np.int32)
    np.savetxt("matriz1.csv", m, fmt="%d", delimiter=",")
    np.savetxt("matriz2.csv", m, fmt="%d", delimiter=",")

    # Executar EXE 1
    (out1, time1) = run_exe(exe1 , tamanho_matriz, seed)

    # Executar EXE 2
    (out2, time2) = run_exe(exe2 , tamanho_matriz, seed)

    if out1 is None or out2 is None:
        print("Erro ao executar arquivos.")
        return

    print("\n======= RESULTADOS =======")
    print(f"{exe1}: {out1} - {time1}s")
    print(f"{exe2}: {out2} - {time2}s")

    print("\n======= COMPARAÇÃO =======")
    if out1 == out2:
        print("As saídas são IGUAIS.")
    else:
        print(f"As saídas são DIFERENTES. {out1} != {out2}")


if __name__ == "__main__":
    main()
