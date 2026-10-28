# ==========================================
# EXPÉRIENCE : Étape 3 - Addition 3 Digits Perfect
# Modèle de base : gpt2 (124M params)
# Hyperparamètres : LR=6e-5, Epochs=6, Batch_Size=16
# Résultat : 100% de réussite sur 100 calculs OOD
# ==========================================


import os
import shutil
import torch
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

class ScratchpadDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=96):
        print(f"Tokenisation du fichier : {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        self.examples = []
        for line in lines:
            tokenized = tokenizer(line, max_length=block_size, truncation=True, return_attention_mask=True)
            self.examples.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"]
            })

    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]

def train_step3():
    model_path = "./gpt2-step2-addition" # On charge les acquis des étapes 1 et 2
    output_path = "./gpt2-step3-addition"
    
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    
    print(f"Chargement du modèle depuis {model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    # Pas besoin de rajouter de jetons spéciaux, ils sont déjà dans le vocabulaire !
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dataset = ScratchpadDataset(tokenizer=tokenizer, file_path="dataset_step3.txt", block_size=96)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=6, # Léger et rapide
        per_device_train_batch_size=16,
        save_strategy="no",  
        logging_steps=50,
        learning_rate=6e-5,
        weight_decay=0.01,
        prediction_loss_only=True,
        logging_dir="./logs"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    print("Début du Fine-Tuning de l'Étape 3 (3 Digits)...")
    trainer.train()
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"🎉 Étape 3 terminée ! Modèle sauvegardé dans : {output_path}")

if __name__ == "__main__":
    train_step3()