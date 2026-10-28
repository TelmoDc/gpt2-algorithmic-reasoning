import torch
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# 1. Création d'un chargeur de données maison avec PyTorch (sans la librairie 'datasets')
class CustomAdditionDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=32):
        self.examples = []
        
        # On lit le fichier texte ligne par ligne
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            
        print(f"Tokenization de {len(lines)} exemples...")
        for line in lines:
            # On tokenise chaque ligne directement
            tokenized = tokenizer(
                line,
                truncation=True,
                max_length=max_length
            )
            self.examples.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"]
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def train_gpt2_step1(dataset_path="dataset_step1.txt", output_dir="./gpt2-step1-addition"):
    print("2. Chargement du tokenizer et du modèle de base...")
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Configuration du token de padding
    tokenizer.pad_token = tokenizer.eos_token

    print("3. Ajout des tokens spéciaux...")
    special_tokens_dict = {'additional_special_tokens': ['[RESULT]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    print("4. Préparation du jeu de données...")
    # On utilise notre classe PyTorch personnalisée à la place de load_dataset
    train_dataset = CustomAdditionDataset(dataset_path, tokenizer, max_length=32)

    # Le data collator s'occupera de mettre toutes les séquences à la même taille (padding)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("5. Configuration des paramètres d'entraînement...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=20,
        per_device_train_batch_size=8,
        save_steps=50,
        save_total_limit=2,
        logging_steps=10,
        learning_rate=5e-5,
        optim="adamw_torch"
    )

    print("6. Lancement de l'entraînement...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    print("7. Sauvegarde du modèle entraîné...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Modèle sauvegardé avec succès dans : {output_dir}")

if __name__ == "__main__":
    train_gpt2_step1()