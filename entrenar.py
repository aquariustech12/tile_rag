from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Cargar modelo base
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-3B",
    max_seq_length = 2048,
    dtype = torch.float16,
    load_in_4bit = True,  # para ahorrar VRAM
)

# Cargar dataset
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

# Configurar entrenamiento
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 100,  # ajusta según necesidad
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "./outputs",
        optim = "adamw_8bit",
        seed = 3407,
    ),
)

trainer.train()

# Guardar modelo fine‑tuned
model.save_pretrained("qwen2.5-3b-finetuned")
tokenizer.save_pretrained("qwen2.5-3b-finetuned")