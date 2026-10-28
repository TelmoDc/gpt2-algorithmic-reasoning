import argparse
import random

def format_number(n):
    return " ".join(list(str(n)))

def generate_step3_dataset(filename="dataset_step3.txt", num_examples=4000):
    dataset = []
    
    print("Génération du dataset Scratchpad Étape 3 (3 chiffres)...")
    for _ in range(num_examples):
        x = random.randint(100, 999)
        y = random.randint(100, 999)
        
        # Décomposition en Unités, Dizaines, Centaines
        x_str, y_str = str(x), str(y)
        x_h, x_t, x_u = int(x_str[0]), int(x_str[1]), int(x_str[2])
        y_h, y_t, y_u = int(y_str[0]), int(y_str[1]), int(y_str[2])
        
        # Cascade des retenues
        u_sum = x_u + y_u
        carry_t = 1 if u_sum >= 10 else 0
        
        t_sum = x_t + y_t + carry_t
        carry_h = 1 if t_sum >= 10 else 0
        
        h_sum = x_h + y_h + carry_h
        
        final_result_spaced = format_number(x + y)
        
        # Chaîne de pensée complète (Scratchpad étendu)
        thought_str = f"[THOUGHT] Units: {x_u} + {y_u} = {u_sum}. Tens: {x_t} + {y_t} + {carry_t} = {t_sum}. Hundreds: {x_h} + {y_h} + {carry_h} = {h_sum}."
        
        # Robustesse des espaces (Chaos control)
        sp1 = " " * random.randint(0, 2)
        sp2 = " " * random.randint(0, 2)
        prompt_math = f"{x}{sp1}+{sp2}{y}".strip()
        
        line = f"{prompt_math} = {thought_str} [RESULT] {final_result_spaced} <|endoftext|>"
        dataset.append(line)
        
    random.shuffle(dataset)
    
    with open(filename, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(item + "\n")
            
    print(f"✅ Fichier '{filename}' créé avec {len(dataset)} exemples.")

if __name__ == "__main__":
    # Par défaut : 4 000 exemples -> dataset_step3.txt (étape 3).
    # Pour l'étude de l'effet de la taille des données :
    #   python data/gen_data_step3.py --n 2000 --out dataset_3digit_train_2000.txt
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4000, help="nombre d'exemples")
    parser.add_argument("--out", default="dataset_step3.txt", help="fichier de sortie")
    args = parser.parse_args()
    generate_step3_dataset(filename=args.out, num_examples=args.n)
