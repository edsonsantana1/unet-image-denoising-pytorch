# Restauração de imagens com U-Net e PyTorch

Implementação de uma U-Net para restaurar imagens aéreas afetadas por ruído gaussiano. O projeto percorre a preparação do dataset, geração de ruído, divisão dos dados, treinamento, validação e avaliação final, com resultados preservados no notebook.

**Autor:** Edson Santana Alves · **Contexto:** atividade acadêmica de Deep Learning

## Resultados

Avaliação no conjunto de teste, com 1.022 imagens, após dez épocas de treinamento:

| **Métrica** | **Resultado** |
| --- | ---: |
| L1 antes da restauração | 0,103727 |
| L1 após a restauração | 0,029693 |
| Redução do erro L1 | 71,37% |
| Parâmetros treináveis | 31.031.875 |
| Dispositivo registrado | CUDA (`cuda:0`) |

O erro L1 médio foi reduzido após a restauração, indicando que o modelo aprendeu a remover parte do ruído adicionado às imagens.

## Implementação

- U-Net com blocos convolucionais e conexões de salto.
- Entrada e saída com três canais e resolução de 256 × 256.
- Ruído gaussiano com `SIGMA = 50`.
- Função de perda L1 (`nn.L1Loss`).
- Otimizador Adam com learning rate `0,0001`.
- Seleção do melhor modelo pela perda de validação.
- Avaliação final separada, sem ajuste de hiperparâmetros no conjunto de teste.
- Semente aleatória `42` para facilitar a reprodução.

## Protocolo experimental

| **Item** | **Configuração** |
| --- | --- |
| Divisão dos dados | treino, validação e teste, com teste de 20% e validação de 15% do restante |
| Tamanho das imagens | 256 × 256 × 3 |
| Batch size | 1 |
| Épocas | 10 |
| Loss | L1 |
| Otimizador | Adam |
| Learning rate | 0,0001 |
| Semente | 42 |

## Executar no VS Code ou Jupyter

Requisitos: Python 3.10 ou superior, acesso à internet para o primeiro download do dataset e, opcionalmente, GPU NVIDIA com CUDA.

No PowerShell, dentro da pasta do projeto:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Abra `atividade_extra.ipynb`, selecione o kernel da pasta `.venv` e execute as células em ordem. O notebook baixa e extrai o dataset automaticamente quando necessário.

O código utiliza CUDA quando disponível e permite execução em CPU. O tempo de treinamento varia conforme o hardware.

## Dataset

O trabalho utiliza o dataset [Forest Aerial Images for Segmentation](https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation). O dataset não é versionado neste repositório; ele é baixado durante a execução do notebook.

## Arquivos gerados

Durante a execução, o notebook pode gerar:

- `unet_melhor_validacao.pth`: pesos do melhor modelo segundo a validação.
- `unet_restauracao.pth`: pesos salvos ao final do treinamento.
- `historico_unet.csv`: perdas de treino e validação por época.

Esses arquivos são ignorados pelo Git para evitar versionar artefatos grandes.

## Limitações

Os resultados correspondem a uma execução com uma única semente e uma configuração de hiperparâmetros. O desempenho pode variar conforme a versão do PyTorch, o hardware e o processo de download ou preparação do dataset. Novos experimentos podem comparar diferentes níveis de ruído, seeds, arquiteturas e métricas como PSNR e SSIM.
