from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
import torch

model_id = "mistralai/Mistral-7B-v0.1"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# Modelo base con soporte 4bit
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)
model = prepare_model_for_kbit_training(model)

# Configuración LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# Cargar dataset
dataset = load_dataset("json", data_files="./data/dataset.jsonl")["train"]

def format(example):
    return {
        "text": f"Instrucción: {example['instruction']}\nEntrada: {example['input']}\nRespuesta: {example['output']}"
    }

dataset = dataset.map(format)

# Configuración de entrenamiento SIN evaluación
training_args = TrainingArguments(
    output_dir="./minerva-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",           # ✅ sin evaluación
    load_best_model_at_end=False,       # ✅ sin modelo "mejor"
    save_total_limit=1,
    bf16=True,
    report_to="none"
)

# Entrenamiento
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=512,
    dataset_text_field="text"
)

trainer.train()

# Guardar modelo y tokenizer
model.save_pretrained("./minerva-lora")
tokenizer.save_pretrained("./minerva-lora")
