import argparse
import os
import shutil
import torch
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# ==========================================
# ⚙️ CONFIGURATION MANUELLE (Modifie ici !)
# ==========================================
TRAIN_FILE = "dataset_3digit_train_2000.txt"  # Remplace par 250, 500, 1000, 2000
# ==========================================

class ScratchpadDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=96):
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

def train_manual():
    model_path = "./gpt2-step2-addition" 
    output_path = "./gpt2-step3-manual" # Écrit par-dessus à chaque nouveau test
    
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    
    print(f"Loading base model from {model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"📖 Lecture du fichier d'entraînement : {TRAIN_FILE}")
    dataset = ScratchpadDataset(tokenizer=tokenizer, file_path=TRAIN_FILE, block_size=96)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=5, 
        per_device_train_batch_size=16,
        save_strategy="no",  
        logging_steps=20,
        learning_rate=6e-5,
        weight_decay=0.01,
        prediction_loss_only=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    print(f"⚡ Début du Fine-Tuning avec {TRAIN_FILE}...")
    trainer.train()
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"🎉 Entraînement terminé ! Modèle sauvegardé dans : {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default=TRAIN_FILE,
                        help="ex : dataset_3digit_train_500.txt (100, 250, 500, 1000, 2000)")
    TRAIN_FILE = parser.parse_args().train_file
    train_manual()
