"""Étape 4 : reclassement (reranking) des meilleurs passages avec un cross-encoder.

1. Les embeddings sélectionnent rapidement les 20 passages les plus proches.
2. Un cross-encoder lit chaque paire (question, passage) en entier et donne un
   score de pertinence plus précis ; on reclasse les 20 candidats avec ce score.
"""

import sys
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from recherche_embeddings import MODELE, charger, vecteurs_passages  # noqa: E402

RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"   # multilingue, pré-entraîné
N_CANDIDATS = 20
K_VALEURS = (1, 3, 5, 10)


def rang_de(classement, passages, cible):
    return next((r for r, i in enumerate(classement, start=1)
                 if (passages[i]["document"], passages[i]["page"]) == cible), None)


def main() -> None:
    # Option : python src/rerank.py models/reranker-hq  -> utilise le reranker spécialisé
    nom_reranker = sys.argv[1] if len(sys.argv) > 1 else RERANKER
    passages, questions = charger()
    modele = SentenceTransformer(MODELE)
    P = vecteurs_passages(modele, passages)
    Q = modele.encode(["query: " + q["question"] for q in questions], normalize_embeddings=True)
    similarites = Q @ P.T

    print(f"Chargement du reranker {nom_reranker}…")
    reranker = CrossEncoder(nom_reranker, max_length=512)

    avant = {k: 0 for k in K_VALEURS}
    apres = {k: 0 for k in K_VALEURS}
    dans_candidats = 0
    print(f"\n{len(questions)} questions, {N_CANDIDATS} candidats reclassés par question\n")
    for q, scores in zip(questions, similarites):
        cible = (q["document"], int(q["page"]))
        candidats = list(np.argsort(-scores)[:N_CANDIDATS])

        paires = [(q["question"], passages[i]["texte"]) for i in candidats]
        scores_rerank = reranker.predict(paires)
        reclasses = [candidats[j] for j in np.argsort(-scores_rerank)]

        r_avant = rang_de(candidats, passages, cible)
        r_apres = rang_de(reclasses, passages, cible)
        dans_candidats += r_avant is not None
        for k in K_VALEURS:
            avant[k] += r_avant is not None and r_avant <= k
            apres[k] += r_apres is not None and r_apres <= k
        fleche = "" if r_avant == r_apres else ("  mieux" if (r_apres or 99) < (r_avant or 99) else "  moins bien")
        print(f"{q['id']}  rang {str(r_avant or '>20'):>4} -> {str(r_apres or '>20'):>4}{fleche}")

    n = len(questions)
    print(f"\nLa bonne page est parmi les {N_CANDIDATS} candidats pour {dans_candidats / n:.1%} des questions")
    print(f"\n{'':28}{'embeddings':>12}{'+ reranker':>12}")
    for k in K_VALEURS:
        print(f"  bonne page dans le top {k:<2}  {avant[k] / n:11.1%} {apres[k] / n:11.1%}")


if __name__ == "__main__":
    main()
