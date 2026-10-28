import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, StoppingCriteria, StoppingCriteriaList

# 1. Condition d'arrêt pour bloquer le bégaiement
class StopOnRepetition(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_text = self.tokenizer.decode(input_ids[0][self.prompt_length:])
        return generated_text.count("[RESULT]") >= 2

def run_benchmark_200():
    # Détection automatique du hardware pour maximiser la vitesse
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = "./gpt2-step3-manual"
    print(f"Chargement du modèle depuis {model_path} sur ({device})...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device).eval()

    # 2. Génération de 200 calculs aléatoires UNIQUES à 3 chiffres
    random.seed(42)  # Même graine pour garder la cohérence des calculs
    test_cases = set()
    while len(test_cases) < 200:
        x = random.randint(100, 999)
        y = random.randint(100, 999)
        test_cases.add((x, y))
    test_cases = list(test_cases)

    success_count = 0
    total_tests = len(test_cases)

    print("\n============================================================")
    print(f"            BENCHMARK DE MASSE : {total_tests} ADDITIONS              ")
    print("============================================================")

    for i, (x, y) in enumerate(test_cases, 1):
        prompt = f"{x} + {y} ="
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = inputs["input_ids"].shape[1]
        
        stopping_criteria = StoppingCriteriaList([StopOnRepetition(tokenizer, prompt_length)])
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=70,  
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                do_sample=False     
            )
        
        raw_reply = tokenizer.decode(outputs[0][prompt_length:])
        
        # Extraction du résultat final après le tag [RESULT]
        try:
            result_part = raw_reply.split("[RESULT]")[1].replace("<|endoftext|>", "").strip()
            predicted_sum = int(result_part.replace(" ", ""))
        except (IndexError, ValueError):
            predicted_sum = -1  # Format cassé ou hallucination
            
        true_sum = x + y
        is_correct = (predicted_sum == true_sum)
        
        if is_correct:
            success_count += 1
            status = "✅"
        else:
            status = "❌"
            
        # Affichage dynamique ligne par ligne
        print(f"[{status}] Test {i:03d}/{total_tests} | {x} + {y} = {true_sum} (IA : {predicted_sum})")

    # Calcul de la précision globale
    accuracy = (success_count / total_tests) * 100

    print("\n============================================================")
    print(f"🎯 ACCURACY FINALE : {success_count}/{total_tests} corrects ({accuracy:.1f}%)")
    print("============================================================")

if __name__ == "__main__":
    run_benchmark_200()