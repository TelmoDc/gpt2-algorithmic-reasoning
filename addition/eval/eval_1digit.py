import random
import torch
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. Chargement du modèle Étape 1
model_path = "./gpt2-step1-addition" 
print("Chargement du modèle Étape 1...")
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).eval()

# 2. Génération des mêmes 100 calculs uniques
random.seed(11) 
test_cases = set()
while len(test_cases) < 100:
    test_cases.add((random.randint(0, 9), random.randint(0, 9)))
test_cases = list(test_cases)

success_count = 0

print("\n============================================================")
print("     CORRECTION BENCHMARK : EXTRACTION SOUPLO-LOGIQUE        ")
print("============================================================")

for i, (x, y) in enumerate(test_cases, 1):
    prompt = f"{x} + {y} ="
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=20,  
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False     
        )
    
    full_reply = tokenizer.decode(outputs[0][prompt_length:])
    true_sum = x + y
    predicted_sum = -1
    
    # Extraction robuste : on vire les espaces et on cherche les premiers chiffres après [RESULT]
    if "[RESULT]" in full_reply:
        suite_result = full_reply.split("[RESULT]")[1]
        # On extrait uniquement les premiers caractères numériques en ignorant les espaces
        chiffres = re.findall(r'\d', suite_result)
        if chiffres:
            predicted_sum = int("".join(chiffres[:2])) # max 2 chiffres pour 1-digit sum

    is_correct = (predicted_sum == true_sum)
    status = "✅" if is_correct else "❌"
    if is_correct: success_count += 1
        
    print(f"[{status}] Test {i:03d}/100 | {x} + {y} = {true_sum} | Bruts IA: {full_reply.strip()} -> Capturé: {predicted_sum}")

accuracy = (success_count / len(test_cases)) * 100
print("\n============================================================")
print(f"🎯 VRAIE ACCURACY DU MODÈLE DIRECT : {accuracy:.1f}%")
print("============================================================")