import random
import json

def generate_scratchpad_tri(numbers):
    current_input = list(numbers)
    sorted_list = []
    thought_lines = []
    
    # Boucle algorithmique : Tri par sélection textuel avec virgules
    while len(current_input) > 0:
        val_min = min(current_input)
        
        # Formatage avec virgules pour stabiliser l'attention
        input_str = ", ".join(map(str, current_input))
        sorted_list_with_min = sorted_list + [val_min]
        sorted_str = ", ".join(map(str, sorted_list_with_min))
        
        line = f"Input: {input_str} | Min: {val_min} | Sorted: {sorted_str}"
        thought_lines.append(line)
        
        # Extraction du minimum
        current_input.remove(val_min)
        sorted_list.append(val_min)
        
    thought_block = "\n".join(thought_lines)
    prompt = ", ".join(map(str, numbers)) + " ="
    full_text = f"{prompt}\n[THOUGHT]\n{thought_block}\n[RESULT] {', '.join(map(str, sorted_list))}"
    
    return {"prompt": prompt, "text": full_text}

# Régénération des 6 000 exemples (longueurs 3 à 5)
random.seed(42)
dataset = []
for _ in range(6000):
    length = random.randint(3, 5)
    numbers = [random.randint(0, 9) for _ in range(length)]
    dataset.append(generate_scratchpad_tri(numbers))

with open("dataset_tri_logique.jsonl", "w", encoding="utf-8") as f:
    for data in dataset:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

print("✅ Nouveau dataset avec virgules généré.")