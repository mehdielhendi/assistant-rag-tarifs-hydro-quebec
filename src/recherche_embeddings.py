"""Étape 3 (deuxième méthode) : recherche de passages par le sens, avec des embeddings.

Un modèle multilingue (multilingual-e5-small) transforme chaque passage et chaque
question en un vecteur de 384 nombres. On classe les passages par similarité
cosinus avec la question, puis on mesure le même score que pour BM25.
"""

import csv
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PASSAGES = Path("data/processed/passages.jsonl")
QUESTIONS = Path("data/questions/questions.csv")
CACHE = Path("data/processed/embeddings_passages.npy")
MODELE = "intfloat/multilingual-e5-small"
K_VALEURS = (1, 3, 5, 10)


def charger():
    passages = [json.loads(l) for l in PASSAGES.open(encoding="utf-8")]
    with QUESTIONS.open(encoding="utf-8") as f:
        questions = [q for q in csv.DictReader(f) if q["reponse"] != "NON_COUVERT"]
    return passages, questions


def vecteurs_passages(modele, passages) -> np.ndarray:
    """Calcule les vecteurs des passages une seule fois, puis les garde en cache."""
    if CACHE.exists():
        return np.load(CACHE)
    # Le modèle e5 attend le préfixe « passage: » devant les textes à chercher
    textes = ["passage: " + p["texte"] for p in passages]
    vecteurs = modele.encode(textes, batch_size=32, normalize_embeddings=True,
                             show_progress_bar=True)
    np.save(CACHE, vecteurs)
    return vecteurs


def main() -> None:
    passages, questions = charger()
    print(f"Chargement du modèle {MODELE} (le premier lancement le télécharge)…")
    modele = SentenceTransformer(MODELE)
    P = vecteurs_passages(modele, passages)

    # ... et le préfixe « query: » devant les questions
    Q = modele.encode(["query: " + q["question"] for q in questions],
                      normalize_embeddings=True)
    similarites = Q @ P.T          # vecteurs normalisés : produit scalaire = cosinus

    trouves = {k: 0 for k in K_VALEURS}
    print(f"\n{len(questions)} questions évaluées sur {len(passages)} passages\n")
    for q, scores in zip(questions, similarites):
        classement = np.argsort(-scores)
        cible = (q["document"], int(q["page"]))
        rang = next((r for r, i in enumerate(classement, start=1)
                     if (passages[i]["document"], passages[i]["page"]) == cible), None)
        for k in K_VALEURS:
            if rang is not None and rang <= k:
                trouves[k] += 1
        meilleur = passages[classement[0]]
        print(f"{q['id']}  rang {rang if rang else '>838':>4}  "
              f"(1er résultat : {meilleur['document']} p.{meilleur['page']})  {q['question'][:60]}")

    print("\nRésultat embeddings (multilingual-e5-small)")
    for k in K_VALEURS:
        print(f"  bonne page dans le top {k:<2} : {trouves[k] / len(questions):6.1%}")


if __name__ == "__main__":
    main()
