"""Étape 3 (première méthode) : recherche de passages par mots-clés avec BM25.

Pour chaque question du jeu de test, on cherche les passages les plus proches
et on vérifie si la bonne page (document + page) est dans les k premiers.
"""

import csv
import json
import re
import unicodedata
from pathlib import Path

import math
from collections import Counter

PASSAGES = Path("data/processed/passages.jsonl")
QUESTIONS = Path("data/questions/questions.csv")
K_VALEURS = (1, 3, 5, 10)

# Mots trop fréquents en français pour aider la recherche
MOTS_VIDES = set("""
le la les un une des de du d l à au aux et ou en pour par sur dans avec ce cet cette ces
qui que quoi quel quelle quels quelles est sont je j me mon ma mes on il elle ils elles
combien comment quand ne pas se s y a
""".split())


def normaliser(texte: str) -> list[str]:
    """Minuscules, sans accents, découpé en mots, sans mots vides."""
    texte = unicodedata.normalize("NFD", texte.lower())
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    mots = re.findall(r"[a-z0-9]+", texte)
    return [m for m in mots if m not in MOTS_VIDES]


class BM25:
    """BM25 écrit à la main : un passage obtient un score élevé s'il contient
    les mots rares de la question, avec une pénalité pour les passages longs."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.freqs = [Counter(d) for d in documents]
        self.longueurs = [len(d) for d in documents]
        self.longueur_moy = sum(self.longueurs) / len(documents)
        n = len(documents)
        df = Counter(mot for d in documents for mot in set(d))
        self.idf = {m: math.log(1 + (n - f + 0.5) / (f + 0.5)) for m, f in df.items()}

    def get_scores(self, requete: list[str]) -> list[float]:
        scores = []
        for freq, longueur in zip(self.freqs, self.longueurs):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * longueur / self.longueur_moy)
            for mot in requete:
                tf = freq.get(mot, 0)
                if tf:
                    s += self.idf[mot] * tf * (self.k1 + 1) / (tf + norm)
            scores.append(s)
        return scores


def charger():
    passages = [json.loads(l) for l in PASSAGES.open(encoding="utf-8")]
    with QUESTIONS.open(encoding="utf-8") as f:
        questions = [q for q in csv.DictReader(f) if q["reponse"] != "NON_COUVERT"]
    return passages, questions


def main() -> None:
    passages, questions = charger()
    bm25 = BM25([normaliser(p["texte"]) for p in passages])

    trouves = {k: 0 for k in K_VALEURS}
    print(f"{len(questions)} questions évaluées sur {len(passages)} passages\n")
    for q in questions:
        scores = bm25.get_scores(normaliser(q["question"]))
        classement = sorted(range(len(passages)), key=lambda i: -scores[i])
        cible = (q["document"], int(q["page"]))
        rang = next((r for r, i in enumerate(classement, start=1)
                     if (passages[i]["document"], passages[i]["page"]) == cible), None)
        for k in K_VALEURS:
            if rang is not None and rang <= k:
                trouves[k] += 1
        meilleur = passages[classement[0]]
        print(f"{q['id']}  rang {rang if rang else '>838':>4}  "
              f"(1er résultat : {meilleur['document']} p.{meilleur['page']})  {q['question'][:60]}")

    print("\nRésultat BM25")
    for k in K_VALEURS:
        print(f"  bonne page dans le top {k:<2} : {trouves[k] / len(questions):6.1%}")


if __name__ == "__main__":
    main()
