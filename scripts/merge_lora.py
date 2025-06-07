from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

# Rutas
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_MODEL = "./minerva-lora"
MERGED_OUTPUT = "./minerva-merged"

# Cargar el modelo LoRA y fusionarlo
model = AutoPeftModelForCausalLM.from_pretrained(
    LORA_MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

model = model.merge_and_unload()

# Guardar el modelo fusionado
model.save_pretrained(MERGED_OUTPUT, safe_serialization=True)

# Guardar el tokenizer original desde el modelo base (¡clave!)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
tokenizer.save_pretrained(MERGED_OUTPUT)

print(f"\n✅ Modelo fusionado guardado en: {MERGED_OUTPUT}")
