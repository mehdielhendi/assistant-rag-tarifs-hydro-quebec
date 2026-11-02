"""Étape 7 : évaluer les réponses générées sur les 50 questions du test.

Mesures automatiques :
  - Hors corpus (6 questions) : le modèle refuse-t-il de répondre ? (sinon = réponse inventée)
  - Couvertes (44 questions)  : refuse-t-il à tort ? cite-t-il la bonne page ?
                                pour les questions de type « chiffre » : les chiffres attendus
                                sont-ils tous dans la réponse ?
Les réponses sont enregistrées dans data/processed/reponses_<modèle>.csv avec une colonne vide
« correct_manuel » : la vérification finale de l'exactitude se fait à la main.
"""

import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from repondre import LLM, REFUS, Assistant  # noqa: E402

RACINE = Path(__file__).resolve().parents[1]
QUESTIONS = RACINE / "data" / "questions" / "questions.csv"
SORTIE = RACINE / "data" / "processed" / f"reponses_{LLM.replace(':', '-')}.csv"   # un fichier par modèle


def est_refus(texte: str) -> bool:
    return "ne trouve pas cette information" in texte.lower()


def chiffres(texte: str) -> set[str]:
    """Extrait les nombres en ignorant les espaces de milliers : « 210 000 » -> « 210000 »."""
    texte = re.sub(r"(?<=\d)[\s  ](?=\d{3}\b)", "", texte)
    return set(re.findall(r"\d+(?:,\d+)?", texte))


def pages_citees(texte: str) -> set[int]:
    return {int(n) for n in re.findall(r"page\s+(\d+)", texte, flags=re.IGNORECASE)}


def main() -> None:
    with QUESTIONS.open(encoding="utf-8") as f:
        questions = list(csv.DictReader(f))
    assistant = Assistant()

    lignes = []
    debut = time.time()
    for q in questions:
        reponse, extraits = assistant.repondre(q["question"])
        hors_corpus = q["reponse"] == "NON_COUVERT"
        ligne = {
            "id": q["id"], "question": q["question"], "reponse_attendue": q["reponse"],
            "reponse_generee": reponse, "refus": est_refus(reponse),
            "page_attendue": q["page"], "pages_citees": " ".join(map(str, sorted(pages_citees(reponse)))),
            "bonne_page_citee": "", "chiffres_ok": "", "correct_manuel": "",
        }
        if not hors_corpus:
            ligne["bonne_page_citee"] = int(q["page"]) in pages_citees(reponse)
            if q["type"] == "chiffre":
                ligne["chiffres_ok"] = chiffres(q["reponse"]) <= chiffres(reponse)
        lignes.append(ligne)
        etat = "REFUS" if ligne["refus"] else "réponse"
        print(f"{q['id']}  {etat:8}  {reponse[:90]!r}")

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    with SORTIE.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)

    hc = [l for l in lignes if l["reponse_attendue"] == "NON_COUVERT"]
    cv = [l for l in lignes if l["reponse_attendue"] != "NON_COUVERT"]
    ch = [l for l in cv if l["chiffres_ok"] != ""]
    pct = lambda n, d: f"{n}/{d} ({n / d:.0%})" if d else "-"  # noqa: E731

    print(f"\n{len(lignes)} questions en {time.time() - debut:.0f} s — réponses dans {SORTIE}\n")
    print("Hors corpus")
    print(f"  refus correct (pas d'invention)   {pct(sum(l['refus'] for l in hc), len(hc))}")
    print("Questions couvertes")
    print(f"  refus à tort                      {pct(sum(l['refus'] for l in cv), len(cv))}")
    print(f"  bonne page citée                  {pct(sum(l['bonne_page_citee'] for l in cv), len(cv))}")
    print(f"  chiffres exacts (type chiffre)    {pct(sum(l['chiffres_ok'] for l in ch), len(ch))}")
    print(f"\nÀ faire à la main : remplir la colonne correct_manuel (1/0) dans {SORTIE.name}")


if __name__ == "__main__":
    main()
