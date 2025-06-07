# Minerva LoRA Fine-Tuning

Este repo contiene el script y dataset para entrenar una versión personalizada del modelo `Mistral-7B` usando LoRA y PEFT.

## Estructura
- `scripts/finetune_minerva.py`: script principal de entrenamiento
- `data/dataset.jsonl`: dataset en formato instruction/output
- `models/`: se crea con los pesos entrenados

## Requisitos
```bash
pip install -r requirements.txt
