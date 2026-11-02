"""Étape 5 : spécialiser (fine-tuning) le cross-encoder sur les documents d'Hydro-Québec.

Données : data/questions/entrainement.csv, 160 questions synthétiques écrites à partir
de passages qui ne sont sur AUCUNE page du jeu de test (vérifié plus bas).

Pour chaque question d'entraînement :
  - 1 paire positive  : (question, son passage)                      -> étiquette 1
  - 4 paires négatives : (question, passages « difficiles »)         -> étiquette 0
    = les passages que les embeddings classent le plus haut, mais sur une autre page.
    Ce sont exactement les erreurs que le reranker doit apprendre à corriger.

La boucle d'entraînement est écrite à la main en PyTorch (perte BCE sur le score du modèle),
pour bien voir ce qui se passe : lot -> score -> perte -> rétropropagation -> mise à jour.
"""

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from recherche_embeddings import MODELE, charger, vecteurs_passages  # noqa: E402
from rerank import RERANKER  # noqa: E402

RACINE = Path(__file__).resolve().parents[1]
ENTRAINEMENT = RACINE / "data" / "questions" / "entrainement.csv"
SORTIE = RACINE / "models" / "reranker-hq"

N_NEGATIFS = 4      # négatifs difficiles par question
EPOQUES = 2         # fixé à l'avance : on ne règle rien en regardant le test
TAILLE_LOT = 16
TAUX = 2e-5         # taux d'apprentissage faible : on ajuste, on ne réapprend pas tout
LONGUEUR_MAX = 384
GRAINE = 42


def main() -> None:
    random.seed(GRAINE)
    torch.manual_seed(GRAINE)

    passages, questions_test = charger()
    index = {p["id"]: i for i, p in enumerate(passages)}
    with open(ENTRAINEMENT, encoding="utf-8") as f:
        entrainement = list(csv.DictReader(f))

    # Garde-fou : aucune question d'entraînement ne doit viser une page du test.
    pages_test = {(q["document"], int(q["page"])) for q in questions_test}
    fuites = [q["id"] for q in entrainement if (q["document"], int(q["page"])) in pages_test]
    if fuites:
        raise SystemExit(f"Fuite vers le jeu de test : {fuites}")
    print(f"{len(entrainement)} questions d'entraînement, aucune sur une page du test")

    # 1. Négatifs difficiles avec les embeddings
    modele = SentenceTransformer(MODELE)
    P = vecteurs_passages(modele, passages)
    Q = modele.encode(["query: " + q["question"] for q in entrainement], normalize_embeddings=True)
    exemples = []
    for q, scores in zip(entrainement, Q @ P.T):
        positif = index[q["passage_id"]]
        page = (q["document"], int(q["page"]))
        exemples.append((q["question"], passages[positif]["texte"], 1.0))
        negatifs = [i for i in np.argsort(-scores)[:20]
                    if (passages[i]["document"], passages[i]["page"]) != page][:N_NEGATIFS]
        exemples += [(q["question"], passages[i]["texte"], 0.0) for i in negatifs]
    print(f"{len(exemples)} paires (1 positive + {N_NEGATIFS} négatives par question)")

    # 2. Boucle d'entraînement
    ce = CrossEncoder(RERANKER, max_length=LONGUEUR_MAX)
    reseau, tokenizer = ce.model, ce.tokenizer
    reseau.train()
    optimiseur = torch.optim.AdamW(reseau.parameters(), lr=TAUX)
    perte_bce = torch.nn.BCEWithLogitsLoss()

    for epoque in range(1, EPOQUES + 1):
        random.shuffle(exemples)
        total = 0.0
        for debut in range(0, len(exemples), TAILLE_LOT):
            lot = exemples[debut:debut + TAILLE_LOT]
            entrees = tokenizer([e[0] for e in lot], [e[1] for e in lot], padding=True,
                                truncation=True, max_length=LONGUEUR_MAX, return_tensors="pt")
            entrees = {k: v.to(reseau.device) for k, v in entrees.items()}
            cibles = torch.tensor([e[2] for e in lot], device=reseau.device)

            scores = reseau(**entrees).logits.view(-1)
            perte = perte_bce(scores, cibles)
            perte.backward()
            optimiseur.step()
            optimiseur.zero_grad()
            total += perte.item() * len(lot)

            n_lot = debut // TAILLE_LOT + 1
            if n_lot % 10 == 0:
                print(f"  époque {epoque}, lot {n_lot}/{-(-len(exemples) // TAILLE_LOT)}, perte {perte.item():.3f}")
        print(f"Époque {epoque} : perte moyenne {total / len(exemples):.3f}")

    # 3. Sauvegarde
    reseau.eval()
    SORTIE.mkdir(parents=True, exist_ok=True)
    reseau.save_pretrained(SORTIE)
    tokenizer.save_pretrained(SORTIE)
    print(f"\nModèle enregistré dans {SORTIE}")
    print("Comparaison sur le test :  python src/rerank.py models/reranker-hq")


if __name__ == "__main__":
    main()
