import random

def format_number(n):
    """Le résultat garde son formatage espacé (ex: 1 3) pour le tokenizer."""
    return " ".join(list(str(n)))

def generate_step_1_robust(filename="dataset_step1.txt"):
    dataset = []
    
    for x in range(10):
        for y in range(10):
            result_str = format_number(x + y)
            
            # On crée plusieurs variantes d'écriture pour le MÊME calcul
            variations = [
                f"{x} + {y} = [RESULT] {result_str} <|endoftext|>",      # Format classique
                f"{x}+{y}=[RESULT] {result_str} <|endoftext|>",          # Format collé
                f"{x}  +  {y}  = [RESULT] {result_str} <|endoftext|>",  # Format très espacé
                f"{x} +{y}  =[RESULT] {result_str} <|endoftext|>",      # Format asymétrique 1
                f"{x}  +{y} = [RESULT] {result_str} <|endoftext|>",      # Format asymétrique 2
            ]
            
            # On ajoute toutes ces variations au dataset
            dataset.extend(variations)
            
    # On mélange le tout
    random.shuffle(dataset)
    
    with open(filename, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(item + "\n")
            
    print(f"Dataset robuste généré : {len(dataset)} exemples (5 variantes pour chaque addition).")

if __name__ == "__main__":
    generate_step_1_robust()