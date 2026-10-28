import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. Chargement du modèle entraîné
MODEL_PATH = "./gpt2-logique-tri"
print(f"Chargement du modèle depuis {MODEL_PATH}...")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

# 2. Cas de test hors distribution (entraînement sur des listes de 3 à 5 éléments)
test_numbers = "5, 8, 2, 1, 9, 4"
prompt = f"{test_numbers} =\n[THOUGHT]"

print("\n" + "="*50)
print(f"📥 PROMPT ENTRÉE (Longueur {len(test_numbers.split(','))}) :\n{test_numbers} =")
print("="*50)

inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 3. Génération avec une marge de tokens plus grande (500)
with torch.no_grad():
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=500,  # Marge large pour toutes les étapes du raisonnement
        do_sample=False,
        temperature=1.0,
        eos_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

print("\n🚀 RAISONNEMENT GÉNÉRÉ PAR GPT-2 :")
print("-" * 50)
print(generated_text)
print("-" * 50)