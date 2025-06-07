from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

# Rutas
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_MODEL = "./minerva-lora"
MERGED_OUTPUT = "./minerva-merged"

# Cargar tokenizer primero (se guarda aparte, no lo toca PEFT)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)

# Cargar modelo LoRA y fusionarlo
model = AutoPeftModelForCausalLM.from_pretrained(
    LORA_MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

model = model.merge_and_unload()

# Guardar modelo fusionado
model.save_pretrained(MERGED_OUTPUT)  # 👈 sin safe_serialization

# Guardar tokenizer desde el modelo base
tokenizer.save_pretrained(MERGED_OUTPUT)

print(f"\n✅ Modelo fusionado guardado en: {MERGED_OUTPUT}")
