import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, StoppingCriteria, StoppingCriteriaList

class StopOnRepetition(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_text = self.tokenizer.decode(input_ids[0][self.prompt_length:])
        return generated_text.count("[RESULT]") >= 2

# 1. Chargement de ton modèle gagnant Étape 3
model_path = "./gpt2-step3-addition" # Remplace par ton dossier de checkpoint à 3 chiffres
print("Chargement du modèle Étape 3 pour le Stress-Test...")
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).eval()

# 2. Génération de 20 cas de test à 4 chiffres (ex: 1234 + 5678)
random.seed(99)
test_cases = []
while len(test_cases) < 20:
    x = random.randint(1000, 9999)
    y = random.randint(1000, 9999)
    test_cases.append((x, y))

print("\n============================================================")
print("          STRESS-TEST : ADDITIONS À 4 CHIFFRES (OOD)        ")
print("============================================================")

success_count = 0

for i, (x, y) in enumerate(test_cases, 1):
    prompt = f"{x} + {y} ="
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_length = inputs["input_ids"].shape[1]
    
    stopping_criteria = StoppingCriteriaList([StopOnRepetition(tokenizer, prompt_length)])
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=90,  # Un peu plus long car il y a un digit de plus à traiter
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            do_sample=False     
        )
    
    full_reply = tokenizer.decode(outputs[0][prompt_length:])
    true_sum = x + y
    
    try:
        result_part = full_reply.split("[RESULT]")[1].replace("<|endoftext|>", "").strip()
        predicted_sum = int(result_part.replace(" ", ""))
    except (IndexError, ValueError):
        predicted_sum = -1
        
    is_correct = (predicted_sum == true_sum)
    status = "✅" if is_correct else "❌"
    if is_correct: success_count += 1
        
    print(f"\n[{status}] Test {i:02d}/20 : {prompt}")
    print(f"Brut généré par l'IA :\n{full_reply.strip()}")
    print(f"Attendu: {true_sum} | Capturé: {predicted_sum}")
    print("-" * 60)

print("\n============================================================")
print(f"📊 BILAN DU STRESS-TEST : {success_count}/20 corrects")
print("============================================================")