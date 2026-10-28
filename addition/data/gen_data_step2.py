import random

def format_number(n):
    return " ".join(list(str(n)))

def generate_step2_dataset(filename="dataset_step2.txt", num_examples=3000):
    dataset = []
    
    print("Génération du dataset Scratchpad Étape 2...")
    for _ in range(num_examples):
        x = random.randint(10, 99)
        y = random.randint(10, 99)
        
        # Extraction des chiffres
        x_str, y_str = str(x), str(y)
        x_t, x_u = int(x_str[0]), int(x_str[1])
        y_t, y_u = int(y_str[0]), int(y_str[1])
        
        # Logique de calcul pas à pas
        u_sum = x_u + y_u
        carry = 1 if u_sum >= 10 else 0
        t_sum = x_t + y_t + carry
        
        final_result_spaced = format_number(x + y)
        
        # Construction de la zone de pensée (Scratchpad)
        thought_str = f"[THOUGHT] Units: {x_u} + {y_u} = {u_sum}. Tens: {x_t} + {y_t} + {carry} = {t_sum}."
        
        # On garde les variations d'espaces de l'Étape 1 pour préserver la robustesse !
        sp1 = " " * random.randint(0, 3)
        sp2 = " " * random.randint(0, 3)
        prompt_math = f"{x}{sp1}+{sp2}{y}".strip()
        
        # Ligne finale du dataset
        line = f"{prompt_math} = {thought_str} [RESULT] {final_result_spaced} <|endoftext|>"
        dataset.append(line)
        
    # Mélange complet
    random.shuffle(dataset)
    
    with open(filename, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(item + "\n")
            
    print(f"✅ Fichier '{filename}' créé avec {len(dataset)} exemples.")

if __name__ == "__main__":
    generate_step2_dataset()