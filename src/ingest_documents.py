"""Télécharge les documents officiels d'Hydro-Québec qui forment le corpus."""

from pathlib import Path

import requests

DOSSIER = Path("data/raw")
BASE = "https://www.hydroquebec.com/data/documents-donnees/pdf/"

# nom local -> fichier sur le site d'Hydro-Québec (versions françaises)
DOCUMENTS = {
    "tarifs-2026.pdf": "tarifs-electricite.pdf",
    "grille-tarifaire-2026.pdf": "grille-tarifaire.pdf",
    "conditions-service-2026.pdf": "conditions-service.pdf",
}


def telecharger() -> None:
    DOSSIER.mkdir(parents=True, exist_ok=True)
    for nom, fichier in DOCUMENTS.items():
        reponse = requests.get(BASE + fichier, timeout=60)
        if reponse.status_code != 200 or not reponse.content.startswith(b"%PDF"):
            print(f"ÉCHEC  {fichier} (code {reponse.status_code})")
            continue
        (DOSSIER / nom).write_bytes(reponse.content)
        print(f"OK     {nom}  {len(reponse.content) / 1e6:.1f} Mo")


if __name__ == "__main__":
    telecharger()
