# Emotion Detection

Este projeto é um sistema de reconhecimento de emoções faciais usando um modelo treinado com o dataset FER-2013.

## Estrutura do Projeto

- `model/`: Contém o modelo treinado (`emotion_model.h5`).
- `data/`: Contém o dataset FER-2013 (`fer2013.csv`).
- `src/`: Contém o código-fonte para treinar o modelo e detectar emoções.
  - `train_model.py`: Código para treinar o modelo.
  - `detect_emotion.py`: Código para detectar emoções em tempo real.
- `requirements.txt`: Lista de bibliotecas necessárias para o projeto.

## Como Usar

### 1. Treinar o Modelo

1. Baixe o dataset FER-2013 e coloque-o no diretório `data/`.
2. Execute o script `train_model.py` para treinar o modelo.

```bash
python src/train_model.py
