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

Requisitos: Python 3.10 ou superior, Git, acesso à internet para o primeiro download do dataset e, opcionalmente, GPU NVIDIA com CUDA. O projeto também pode ser executado somente em CPU, mas o treinamento será mais demorado.

### Clonar o repositório

```powershell
git clone https://github.com/edsonsantana1/unet-image-denoising-pytorch.git
cd unet-image-denoising-pytorch
```

### Criar o ambiente

No PowerShell, dentro da pasta do projeto:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Abra `atividade_extra.ipynb`, selecione o kernel da pasta `.venv` e execute as células em ordem. O notebook baixa e extrai o dataset automaticamente quando necessário.

O código utiliza CUDA quando disponível e permite execução em CPU. O tempo de treinamento varia conforme o hardware. Para abrir o notebook pelo terminal, use:

```powershell
.\.venv\Scripts\jupyter.exe notebook atividade_extra.ipynb
```

Para configurar CUDA, consulte o [seletor oficial de instalação do PyTorch](https://pytorch.org/get-started/locally/). A instalação das dependências deste projeto não garante, sozinha, uma versão do PyTorch com suporte a CUDA.

### Solução de problemas

- **`python` ou `py` não encontrado:** instale o Python e marque a opção de adicionar o Python ao PATH; depois abra um novo terminal.
- **O PowerShell bloqueia a ativação do ambiente:** execute `Set-ExecutionPolicy -Scope Process Bypass` e tente novamente.
- **Jupyter não aparece no VS Code:** instale as extensões Python e Jupyter, selecione o interpretador de `.venv` e reinicie o kernel.
- **CUDA indisponível:** o notebook pode usar CPU. Para verificar o dispositivo selecionado, observe a saída da primeira célula.
- **Falha no download do dataset:** confirme o acesso à internet e execute novamente a célula de download. O dataset é extraído na pasta `kaggle_forest_segmentation/`.
- **Memória insuficiente:** mantenha `BATCH_SIZE = 1` e feche outros programas que estejam usando a GPU.

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
