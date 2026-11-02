"""Extrait le texte des PDF et le découpe en passages pour la recherche.

Chaque passage garde son document et sa page, pour que chaque réponse
puisse citer sa source.
"""

import json
import re
from pathlib import Path

import pdfplumber

ENTREE = Path("data/raw")
SORTIE = Path("data/processed/passages.jsonl")
TAILLE = 180        # mots par passage
CHEVAUCHEMENT = 40  # mots repris du passage précédent


def nettoyer(texte: str) -> str:
    texte = texte.replace("­", "")              # traits d'union invisibles
    texte = re.sub(r"-\n(\w)", r"\1", texte)         # mots coupés en fin de ligne
    return re.sub(r"\s+", " ", texte).strip()


def decouper_page(mots: list[str]) -> list[str]:
    pas = TAILLE - CHEVAUCHEMENT
    return [" ".join(mots[i:i + TAILLE]) for i in range(0, max(len(mots) - CHEVAUCHEMENT, 1), pas)]


def main() -> None:
    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with SORTIE.open("w", encoding="utf-8") as f:
        for pdf_path in sorted(ENTREE.glob("*.pdf")):
            n_doc = 0
            with pdfplumber.open(pdf_path) as pdf:
                for num, page in enumerate(pdf.pages, start=1):
                    mots = nettoyer(page.extract_text() or "").split()
                    if len(mots) < 20:      # page vide, titre seul ou table des matières courte
                        continue
                    for k, texte in enumerate(decouper_page(mots)):
                        passage = {
                            "id": f"{pdf_path.stem}-p{num}-{k}",
                            "document": pdf_path.stem,
                            "page": num,
                            "texte": texte,
                        }
                        f.write(json.dumps(passage, ensure_ascii=False) + "\n")
                        n_doc += 1
            print(f"{pdf_path.stem:28s} {n_doc:5d} passages")
            total += n_doc
    print(f"Total : {total} passages -> {SORTIE}")


if __name__ == "__main__":
    main()
