import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, StoppingCriteria, StoppingCriteriaList

# 1. Condition d'arrêt intelligente pour éviter le bégaiement
class StopOnRepetition(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_text = self.tokenizer.decode(input_ids[0][self.prompt_length:])
        return generated_text.count("[RESULT]") >= 2

# 2. Chargement du modèle Étape 3
model_path = "./gpt2-step3-addition"
print("Chargement du modèle final...")
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).eval()

# 3. Le Dataset de Test Ultime (Mélange de toutes les étapes)
tests = [
    # --- ÉTAPE 1 (1 digit) ---
    "3 + 5",
    "9+9",
    
    # --- ÉTAPE 2 (2 digits) ---
    "27 + 35",
    "58+24",
    
    # --- ÉTAPE 3 (3 digits) ---
    "123 + 456",
    "789 + 456",
    "999+999"
]

print("\n============================================================")
print("             GRAND CRASH TEST MULTI-FORMATS                 ")
print("============================================================")

for calcul in tests:
    prompt = f"{calcul} ="
    inputs = tokenizer(prompt, return_tensors="pt")
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
    
    # Nettoyage propre de la sortie
    try:
        clean_reply = raw_reply.split("[RESULT]")[0] + "[RESULT]" + raw_reply.split("[RESULT]")[1]
    except IndexError:
        clean_reply = raw_reply
        
    clean_reply = clean_reply.replace("<|endoftext|>", "").strip()
    
    print(f"📥 Calcul : {calcul}")
    print(f"🤖 Sortie : {clean_reply}")
    print("-" * 60)