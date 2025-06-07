from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_MODEL = "./minerva-lora"
MERGED_OUTPUT = "./minerva-merged"

# Cargar el modelo LoRA y fusionarlo
model = AutoPeftModelForCausalLM.from_pretrained(
    LORA_MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    tokenizer=None  # ⚠️ importante: evitar que lo busque en LORA
)

model = model.merge_and_unload()

# Guardar el modelo fusionado
model.save_pretrained(MERGED_OUTPUT, safe_serialization=True)

# Usar tokenizer del modelo base
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
tokenizer.save_pretrained(MERGED_OUTPUT)

print(f"\n✅ Modelo fusionado guardado en: {MERGED_OUTPUT}")
