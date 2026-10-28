import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# 1. Initialisation du modèle et du Tokenizer
MODEL_NAME = "gpt2" # On repart de la base brute pour le tri
print(f"Initialisation de {MODEL_NAME}...")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# Configuration du token de padding
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# 2. Chargement et tokenisation du dataset de tri
print("Chargement du dataset de tri...")
dataset = load_dataset("json", data_files={"train": "dataset_tri_logique.jsonl"})

def tokenize_function(examples):
    # On tokenize le bloc textuel complet. max_length à 256 pour éviter toute coupure du raisonnement
    outputs = tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "text"])

# Split train/validation (90% / 10%)
split_dataset = tokenized_datasets["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# 3. Configuration optimisée pour la mémoire de l'iMac (MPS)
training_args = TrainingArguments(
    output_dir="./gpt2-logique-tri",
    num_train_epochs=3,                 
    per_device_train_batch_size=16,     # Baissé de 32 à 16 pour éviter l'Out-Of-Memory
    gradient_accumulation_steps=4,      # Augmenté à 4 pour maintenir un gros batch virtuel (16 x 4 = 64)
    per_device_eval_batch_size=16,      # Baissé aussi pour l'évaluation
    eval_strategy="epoch",       
    save_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=20,                   
    save_total_limit=1,                 
    fp16=False,                         # Note : Garde False sur Mac, MPS gère mal le fp16 standard de Hugging Face
    report_to="none"
)

# 4. Lancement du Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

print("🚀 Début de l'entraînement du modèle sur l'algorithme de tri...")
trainer.train()

# Sauvegarde finale du modèle aligné
trainer.save_model("./gpt2-logique-tri")
tokenizer.save_pretrained("./gpt2-logique-tri")
print("🎯 Entraînement terminé et modèle sauvegardé dans './gpt2-logique-tri' !")