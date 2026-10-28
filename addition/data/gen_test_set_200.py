import random

# Utilisation de la graine 42 pour être cohérent avec ton premier script
random.seed(42)

# Génération de 200 calculs uniques
test_cases = set()
while len(test_cases) < 200:
    x = random.randint(100, 999)
    y = random.randint(100, 999)
    test_cases.add((x, y))

# Tri pour avoir un fichier propre et minutieux
dataset_lines = [f"{x} + {y} = {x + y}" for x, y in sorted(test_cases)]

# Écriture du fichier txt
with open("dataset_3digit_random_200.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(dataset_lines))

print("✅ Le fichier 'dataset_3digit_random_200.txt' a été créé avec succès !")