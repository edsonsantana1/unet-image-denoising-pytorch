# Restauração de imagens com U-Net e PyTorch

Trabalho de aprendizado de máquina e profundo para remoção de ruído gaussiano em imagens usando uma arquitetura U-Net.

## Conteúdo

- `atividade_extra.ipynb`: notebook completo do trabalho.
- `requirements.txt`: dependências principais.

## Como executar

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook atividade_extra.ipynb
```

O notebook baixa o dataset quando necessário. O treinamento foi preparado para utilizar CUDA quando uma GPU NVIDIA compatível estiver disponível.

## Dataset

O trabalho utiliza o dataset [Forest Aerial Images for Segmentation](https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation). Os arquivos do dataset não são versionados neste repositório.
