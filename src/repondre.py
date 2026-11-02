"""Étape 6 : générer une réponse à partir des passages trouvés (RAG complet).

question -> embeddings (20 candidats) -> reranker (3 meilleurs) -> LLM -> réponse citée

LLM : Ollama en local (qwen2.5:7b) ; en ligne, l'API Groq si la variable GROQ_API_KEY existe.

Utilisation :  python src/repondre.py "Combien coûte le kWh au tarif D ?"
Prérequis    :  Ollama installé et  ollama pull qwen2.5:7b
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from recherche_embeddings import MODELE, charger, vecteurs_passages  # noqa: E402
from rerank import N_CANDIDATS, RERANKER  # noqa: E402

LLM = "qwen2.5:7b"
OLLAMA = "http://localhost:11434/api/chat"
N_CONTEXTE = 3                     # passages donnés au LLM
# En ligne (Streamlit Cloud), pas d'Ollama : si une clé GROQ_API_KEY est définie,
# on appelle l'API gratuite de Groq (format compatible OpenAI) au lieu du modèle local.
GROQ = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELE = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
REFUS = "Je ne trouve pas cette information dans les documents d'Hydro-Québec."

CONSIGNE = f"""Tu es un assistant qui répond aux questions sur les tarifs et les conditions de service d'Hydro-Québec.
Règles :
1. Réponds en français, en 1 à 3 phrases, en utilisant UNIQUEMENT les extraits fournis.
2. Recopie les montants et les chiffres exactement comme dans les extraits.
3. Termine par la source sous la forme : (Source : <document>, page <numéro>).
4. Si les extraits ne contiennent pas la réponse, réponds exactement : {REFUS}
N'utilise jamais tes connaissances générales."""


class Assistant:
    def __init__(self) -> None:
        self.passages, _ = charger()
        self.encodeur = SentenceTransformer(MODELE)
        self.P = vecteurs_passages(self.encodeur, self.passages)
        self.reranker = CrossEncoder(RERANKER, max_length=512)

    def classer(self, question: str) -> list[dict]:
        """Les 20 candidats des embeddings, reclassés par le reranker, avec les deux rangs et scores."""
        q = self.encodeur.encode(["query: " + question], normalize_embeddings=True)[0]
        similarites = self.P @ q
        candidats = np.argsort(-similarites)[:N_CANDIDATS]
        scores = self.reranker.predict([(question, self.passages[i]["texte"]) for i in candidats])
        classement = []
        for rang_rerank, j in enumerate(np.argsort(-scores), start=1):
            i = candidats[j]
            classement.append({**self.passages[i],
                               "rang_embeddings": int(j) + 1, "similarite": float(similarites[i]),
                               "rang_reranker": rang_rerank, "pertinence": float(scores[j])})
        return classement

    def chercher(self, question: str, n: int = N_CONTEXTE) -> list[dict]:
        return self.classer(question)[:n]

    def repondre(self, question: str) -> tuple[str, list[dict]]:
        extraits = self.chercher(question)
        return appeler_llm(construire_messages(question, extraits)), extraits


def construire_messages(question: str, extraits: list[dict]) -> list[dict]:
    contexte = "\n\n".join(f"[Extrait {n} — {p['document']}, page {p['page']}]\n{p['texte']}"
                           for n, p in enumerate(extraits, start=1))
    return [{"role": "system", "content": CONSIGNE},
            {"role": "user", "content": f"Extraits :\n{contexte}\n\nQuestion : {question}"}]


def nom_modele() -> str:
    return f"{GROQ_MODELE} (Groq)" if os.environ.get("GROQ_API_KEY") else f"{LLM} (Ollama, local)"


def _requete(messages: list[dict], flux: bool) -> requests.Response:
    cle = os.environ.get("GROQ_API_KEY")
    if cle:
        r = requests.post(GROQ, timeout=60, stream=flux, headers={"Authorization": f"Bearer {cle}"},
                          json={"model": GROQ_MODELE, "temperature": 0, "stream": flux, "messages": messages})
    else:
        options = {"temperature": 0}            # même question -> même réponse
        corps = {"model": LLM, "stream": flux, "options": options, "messages": messages}
        r = requests.post(OLLAMA, timeout=300, stream=flux, json=corps)
        if r.status_code == 500 and "CUDA" in r.text:
            # La carte graphique (4 Go) refuse parfois de charger le modèle : on réessaie sur le processeur
            options["num_gpu"] = 0
            r = requests.post(OLLAMA, timeout=600, stream=flux, json=corps)
        if r.status_code == 404:
            raise RuntimeError(f"Le modèle {LLM} n'est pas installé dans Ollama : lancez  ollama pull {LLM}")
    if not r.ok:                                    # afficher le vrai message du serveur
        raise RuntimeError(f"Le serveur du modèle a répondu {r.status_code} : {r.text[:300]}")
    return r


def appeler_llm(messages: list[dict]) -> str:
    r = _requete(messages, flux=False)
    donnees = r.json()
    if "choices" in donnees:                        # format Groq / OpenAI
        return donnees["choices"][0]["message"]["content"].strip()
    return donnees["message"]["content"].strip()


def appeler_llm_flux(messages: list[dict]):
    """Renvoie la réponse morceau par morceau, pour l'afficher pendant qu'elle s'écrit."""
    r = _requete(messages, flux=True)
    for ligne in r.iter_lines(decode_unicode=True):
        if not ligne:
            continue
        if ligne.startswith("data: "):              # format Groq / OpenAI (server-sent events)
            ligne = ligne[6:]
            if ligne == "[DONE]":
                break
            yield json.loads(ligne)["choices"][0]["delta"].get("content") or ""
        else:                                       # format Ollama (une ligne JSON par morceau)
            yield json.loads(ligne).get("message", {}).get("content", "")


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Combien coûte le kWh au tarif D ?"
    assistant = Assistant()
    reponse, extraits = assistant.repondre(question)
    print(f"\nQuestion : {question}\n\nRéponse : {reponse}\n\nPassages utilisés :")
    for p in extraits:
        print(f"  - {p['document']}, page {p['page']}  (pertinence {p['pertinence']:.2f})")


if __name__ == "__main__":
    main()
