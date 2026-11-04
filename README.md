# Paralelização de processamento de imagens com OpenMP

Projeto de TCC que compara programas sequenciais com versões paralelizadas em C++.

Os programas avaliados são Threshold, Equalização de Histograma e Sobel–Prewitt–Roberts. Os três detectores permanecem juntos em um executável; o Threshold calcula suas cinco modalidades em cada execução.

## Organização

| Pasta | Conteúdo |
| --- | --- |
| originais/ | Referências sequenciais adaptadas para execução sem interface gráfica |
| paralelizados/ | Versões com OpenMP |
| originais/common/ e paralelizados/common/ | Leitura, gravação e temporização de cada versão |
| scripts/ | Comandos de compilação, verificação e coleta |
| tools/ | Coleta em Python, preparo das imagens e diagnóstico de afinidade |
| testes/ | Comparação das saídas e testes do protocolo |
| tcc-images/ | Imagens utilizadas na coleta e nos testes |
| referencia-arce/ | Código histórico usado como base, incluindo os outros grupos do estudo |
| resultados/ | Medições e informações do ambiente |

A referência sequencial contém adaptações de execução e correções comuns: Trunc com limiar 80, inclusão dos 256 níveis na PMF/CDF e normalização constante em zero. Ela não é uma cópia intocada do código histórico.

O ganho frente à versão paralelizada inclui as refatorações utilizadas, além da execução com OpenMP.

## Dependências

| Dependência | Requisito |
| --- | --- |
| Sistema operacional | Linux |
| Shell | Bash |
| Python | Versão 3.9 ou superior |
| Fuso horário | America/Sao\_Paulo |
| Compilador | G++ com suporte a OpenMP e C++17 |
| Gerenciador de pacotes | pkg-config |
| Biblioteca de processamento de imagens | OpenCV 4 (desenvolvimento) |
| Controle de afinidade | taskset |
| Informações da CPU | lscpu |
| Medição de tempo | GNU time em `/usr/bin/time` |
| Perf | Necessário apenas no modo `official` |
| Permissões | Acesso aos contadores de desempenho, necessário apenas no modo `official` |

## Execução

Na raiz do repositório:

bash

```bash
./scripts/compilar.sh
./scripts/verificar.sh
./scripts/benchmark.sh
```

A compilação cria seis executáveis em bin/sequenciais/ e bin/paralelos/. A verificação testa 44 entradas, incluindo casos sintéticos, matrizes não contíguas e intermediários float. O benchmark compila seus próprios executáveis temporários; a compilação manual não é um pré-requisito.

| Modo do benchmark | Comportamento |
| --- | --- |
| local-ht (padrão) | Permite HT/SMT e mede 1, 2, 4, 6 e 8 threads, se houver CPUs lógicas suficientes |
| local | Limita os tempos aos valores que cabem nos núcleos físicos disponíveis |
| official | Exige oito núcleos físicos disponíveis, SMT inativo e acesso ao perf |
| verify | Executa a validação completa, incluindo 8K, sem coletar tempos de desempenho |

bash

```bash
./scripts/benchmark.sh --mode verify
./scripts/benchmark.sh --mode official
```

No laptop de quatro núcleos físicos, seis e oito threads compartilham núcleos. O script registra a topologia e verifica a afinidade; não altera configurações do sistema. Na infraestrutura da faculdade, utilizar a chococino somente para acessar uma máquina de processamento autorizada.

## Protocolo da coleta

São usadas Lena, Sudoku e Building em 500×500, 1024×1024, 2048×2048, 3840×2160 e 7680×4320. As entradas são geradas em cinza de oito bits, com INTER\_AREA e armazenamento PNG. A entrada 8K é redimensionada da fonte.

Cada combinação de programa, imagem, resolução e número de threads tem um aquecimento descartado e três medições. Para o sequencial, há uma única configuração por programa/imagem/resolução, também com um aquecimento e três medições; ele não é repetido para cada número de threads. A bateria completa tem 270 configurações: 270 aquecimentos e 810 medições, totalizando 1.080 execuções dos programas. As verificações de corretude são executadas separadamente.

O tempo principal é o intervalo interno do núcleo, excluindo leitura e gravação de imagens e alocação das matrizes principais. As flags comuns são -O3 -std=c++17 -ffp-contract=off; só o paralelo recebe -fopenmp. O OpenCV utiliza uma thread e o OpenCL fica desativado.

Na coleta local, as medições são aleatorizadas com semente registrada. No modo official, perf stat -d -r 3 executa as três repetições por bloco. GNU time e perf medem o processo completo, não apenas o núcleo.

## Resultados

Cada coleta de desempenho salva dois arquivos:

- results.jsonl: aquecimentos e medições, com imagem, resolução, variante, threads, tempo em segundos e memória máxima.
- environment.json: máquina, versões, flags, hashes, afinidade e situação da execução.

Para analisar os tempos, selecionar phase igual a measured. Nas novas coletas, process\_reports guarda os relatórios de GNU time e perf. O relatório perf é agregado sobre três execuções e aparece apenas na primeira amostra do bloco. Em verify, é salvo somente environment.json.

Binários, entradas redimensionadas e imagens de validação são temporários e removidos ao terminar. As medições anteriores são preservadas.


## Testes do protocolo

bash

```bash
python3 -B -m unittest discover -s testes -p 'test_*.py'
```

Esses testes conferem o parsing dos tempos, as repetições e as regras de threads e afinidade. Eles complementam a verificação das imagens.