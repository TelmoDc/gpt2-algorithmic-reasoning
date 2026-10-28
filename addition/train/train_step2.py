import torch
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import os
import shutil
# On remplace TextDataset par une classe PyTorch native, bien plus stable
class ScratchpadDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=64):
        print(f"Chargement et tokenisation du fichier : {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            # On récupère les lignes non vides
            lines = [line.strip() for line in f if line.strip()]
        
        self.examples = []
        for line in lines:
            # Tokenisation de la ligne sans padding (le DataCollator s'en chargera dynamiquement)
            tokenized = tokenizer(
                line,
                max_length=block_size,
                truncation=True,
                return_attention_mask=True
            )
            self.examples.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"]
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def train_step2():
    model_path = "./gpt2-step1-addition" 
    output_path = "./gpt2-step2-addition"
    
    # Nettoyage manuel du dossier de sortie s'il existe déjà (remplace 'overwrite_output_dir')
    if os.path.exists(output_path):
        print(f"Nettoyage de l'ancien dossier de sauvegarde : {output_path}")
        shutil.rmtree(output_path)
    
    print(f"Chargement du modèle depuis {model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    special_tokens_dict = {'additional_special_tokens': ['[THOUGHT]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dataset = ScratchpadDataset(
        tokenizer=tokenizer,
        file_path="dataset_step2.txt",
        block_size=64
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Configuration de l'entraînement (sans l'argument problématique)
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=5, 
        per_device_train_batch_size=16,
        save_strategy="no",  
        logging_steps=100,
        learning_rate=5e-5,
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
    
    print("Début du Fine-Tuning de l'Étape 2 (Scratchpad)...")
    trainer.train()
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"🎉 Étape 2 bouclée avec succès ! Modèle sauvegardé dans : {output_path}")
if __name__ == "__main__":
    train_step2()