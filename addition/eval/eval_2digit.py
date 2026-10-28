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

# 1. Chargement du modèle Étape 2
model_path = "./gpt2-step2-addition" 
print("Chargement du modèle Étape 2...")
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).eval()

# 2. Génération de 100 calculs uniques à 2 chiffres
random.seed(22) 
test_cases = set()
while len(test_cases) < 100:
    test_cases.add((random.randint(10, 99), random.randint(10, 99)))
test_cases = list(test_cases)

success_count = 0

print("\n============================================================")
print("        AFFICHAGE DES 100 RÉPONSES DÉTAILLÉES (2-DIGITS)    ")
print("============================================================")

for i, (x, y) in enumerate(test_cases, 1):
    prompt = f"{x} + {y} ="
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_length = inputs["input_ids"].shape[1]
    
    stopping_criteria = StoppingCriteriaList([StopOnRepetition(tokenizer, prompt_length)])
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=55,  
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
        
    if predicted_sum == true_sum:
        success_count += 1
        status = "✅"
    else:
        status = "❌"

    # Affichage de la trace complète générée par l'IA
    print(f"\n[{status}] Opération {i:03d}/100 : {prompt}")
    print(full_reply.strip())
    print("-" * 50)

accuracy = (success_count / len(test_cases)) * 100
print("\n============================================================")
print(f"🎯 ACCURACY FINALE 2-DIGITS : {success_count}/100 corrects ({accuracy:.1f}%)")
print("============================================================")