
import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def run_interactive_benchmark():
    # Détection de la puce (MPS pour Mac, sinon CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_path = "./gpt2-logique-tri"  # À adapter avec ton vrai dossier de modèle
    
    print(f"Chargement du modèle sur ({device})...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device).eval()
    
    random.seed(42) # Pour la cohérence, mais tu peux l'enlever pour du full aléatoire

    while True:
        print("\n============================================================")
        print("          MODE VÉRIFICATION MANUELLE (UN PAR UN)            ")
        print("============================================================")
        
        choix_taille = input("👉 Quelle longueur de liste veux-tu tester ? (ou 'q' pour quitter) : ")
        if choix_taille.lower() == 'q':
            print("Fermeture du script. Bon débuggage !")
            break
            
        if not choix_taille.isdigit():
            print("❌ Erreur : Entre un nombre entier valide.")
            continue
            
        longueur = int(choix_taille)
        
        choix_nb = input(f"👉 Combien de listes de taille {longueur} veux-tu inspecter ? [Défaut: 5] : ")
        nb_tests = int(choix_nb) if choix_nb.strip().isdigit() else 5
        
        print(f"\n🚀 Lancement de {nb_tests} tests pour la taille {longueur}...")
        print("Le script va s'interrompre après chaque réponse. Regarde bien le résultat.")
        
        for t in range(1, nb_tests + 1):
            # 1. Génération de la liste de chiffres uniques
            liste_chiffres = random.sample(range(0, 10), longueur)
            prompt = ", ".join([str(n) for n in liste_chiffres]) + " ="
            
            # 2. Inférence du modèle
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_length = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=250, # Assez de tokens pour le scratchpad si présent
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )
            
            # Décodage complet (on garde tout le texte généré après le =)
            reponse_ia = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=False).strip()
            
            # 3. Affichage épuré pour ton inspection
            print(f"\n🔍 [TEST {t:02d}/{nb_tests} - Longueur {longueur}]")
            print(f"📥 Entrée : {prompt}")
            print(f"🤖 Sortie : {reponse_ia}")
            print("-" * 40)
            
            # 4. Blocage et attente de l'utilisateur
            if t < nb_tests:
                input("⌨️ Appuie sur [ENTRÉE] pour voir le test suivant...")
            else:
                input("🏁 Fin de la série. Appuie sur [ENTRÉE] pour revenir au menu principal...")

if __name__ == "__main__":
    run_interactive_benchmark()