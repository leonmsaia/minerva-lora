from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Rutas
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_MODEL = "./minerva-lora"
MERGED_OUTPUT = "./minerva-merged"

# Cargar modelo base primero (opcional en esta variante, se usa por fallback interno)
# base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)

# Cargar modelo LoRA fusionado
model = AutoPeftModelForCausalLM.from_pretrained(
    LORA_MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

# Fusionar con modelo base y descargar capas LoRA
model = model.merge_and_unload()

# Guardar el modelo fusionado
model.save_pretrained(MERGED_OUTPUT, safe_serialization=True)

# Guardar tokenizer del modelo base
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
tokenizer.save_pretrained(MERGED_OUTPUT)

print(f"\n✅ Modelo fusionado guardado en: {MERGED_OUTPUT}")
