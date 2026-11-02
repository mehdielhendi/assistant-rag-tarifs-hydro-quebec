# Assistant RAG sur les tarifs d'électricité d'Hydro-Québec

![Python](https://img.shields.io/badge/Python-3-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-sentence--transformers-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-app-ff4b4b)
![Évalué](https://img.shields.io/badge/évalué%20sur-50%20questions-brightgreen)

Assistant qui répond en français aux questions d'un client sur les tarifs et les conditions de service
d'Hydro-Québec. Il répond **uniquement** à partir des documents officiels, cite le document et la page,
et dit « je ne trouve pas » plutôt que d'inventer. Chaque étape est mesurée sur un jeu de 50 questions
annoté à la main.

**En bref : 84 % de réponses correctes (vérifiées une par une), 0 réponse inventée sur 6 questions
hors corpus, bonne page dans les 3 premiers passages pour 84 % des questions.**

```
question ─► embeddings e5-small (838 passages → 20) ─► reranker cross-encoder (20 → 3) ─► LLM ─► réponse sourcée ou refus
```

Projet personnel d'El Hendi Mohamed Mehdi (maîtrise en informatique, IA et science des données,
Université de Sherbrooke). Non affilié à Hydro-Québec.

## Application

`streamlit run app.py` ouvre une application en trois onglets :

- **Assistant** : conversation, réponse affichée pendant qu'elle s'écrit, source cliquable qui ouvre le PDF
  officiel à la bonne page, score de pertinence de chaque passage, et option pour voir comment le reranker
  a reclassé les 20 candidats des embeddings.
- **Évaluation** : indicateurs, comparaison BM25 / embeddings / reranker, comparaison de deux LLM,
  et les 50 réponses vérifiées, filtrables.
- **Méthode** : schéma du pipeline et liens vers les documents sources.

Le LLM tourne en local avec [Ollama](https://ollama.com) (`qwen2.5:7b`). Si la variable d'environnement
`GROQ_API_KEY` est définie, l'application utilise plutôt l'API de Groq (utile pour un déploiement sans GPU).

## Résultats (recherche de la bonne page)

Jeu de test : 50 questions formulées comme un client les poserait (44 avec une réponse dans les
documents, 6 hors corpus), chacune avec sa réponse et sa page source vérifiées dans les PDF.
Corpus : 3 documents officiels d'Hydro-Québec (tarifs 2026, grille tarifaire, conditions de service),
découpés en 838 passages.

| Méthode | Top 1 | Top 3 | Top 5 | Top 10 |
| --- | --- | --- | --- | --- |
| BM25 (mots-clés, implémenté à la main) | 23 % | 50 % | 66 % | 73 % |
| Embeddings `multilingual-e5-small` | 45 % | 73 % | 84 % | 91 % |
| Embeddings + reranker `mmarco-mMiniLMv2` (20 candidats) | **64 %** | **84 %** | **91 %** | **93 %** |

**Ce qu'on apprend.** BM25 échoue sur les codes courts (« tarif D ») et sur les reformulations
(« me couper l'électricité » contre « interrompre le service »). Les embeddings comparent le sens et
corrigent la plupart de ces cas. Le reranker, qui lit la question et le passage ensemble, fait passer
la bonne page en première position pour 64 % des questions.

**Essai de spécialisation du reranker (résultat négatif).** J'ai entraîné le reranker en PyTorch
sur 160 questions synthétiques tirées uniquement de pages absentes du test, avec des négatifs
« difficiles » (passages bien classés par les embeddings mais sur une autre page). Les réglages
(2 époques, taux 2e-5) ont été fixés avant de regarder le test. Résultat : le top 1 passe de 64 % à 57 %
et le top 3 de 84 % à 77 %. Causes probables : faux négatifs (les documents répètent des clauses presque
identiques d'une section à l'autre), questions synthétiques plus proches du texte que celles d'un client,
et trop peu d'exemples. Je garde donc le reranker pré-entraîné.

## Résultats (réponse générée)

Les 3 meilleurs passages sont donnés à un LLM local (Ollama) avec la consigne de répondre seulement
à partir de ces extraits, de citer la page, et de refuser si la réponse n'y est pas.

| Modèle | Refus correct hors corpus (6) | Refus à tort (44) | Bonne page citée | Chiffres exacts (19) |
| --- | --- | --- | --- | --- |
| `qwen2.5:3b` | 6/6 | 16/44 (36 %) | 59 % | 37 % |
| `qwen2.5:7b` | **6/6** | **4/44 (9 %)** | **73 %** | **58 %** |

**Exactitude du 7B, vérifiée réponse par réponse contre les PDF : 37/44 (84 %).** Sur les 7 erreurs,
4 sont des refus à tort et 3 sont des réponses fausses (7 %), dont une confusion de prix au tarif DT.
La relecture a aussi révélé 2 réponses de référence incomplètes dans le jeu de test (q026, q046) :
le modèle avait raison selon le texte.

Le 3B refuse beaucoup, donc son 100 % hors corpus ne prouve rien ; le 7B refuse rarement à tort et
n'invente toujours rien sur les questions hors corpus. Les erreurs restantes du 7B sont surtout des
confusions entre deux montants proches (prix de la puissance contre prix de l'énergie, deux sections
du tarif DT). Le test « chiffres exacts » est strict : il exige tous les nombres de la réponse de référence.

**Limite.** 44 questions évaluées : une question vaut environ 2,3 points de pourcentage.

## Étapes

1. `python src/ingest_documents.py` : télécharge les documents officiels
2. `python src/decouper.py` : extrait le texte et le découpe en passages (document + page)
3. `python src/recherche_bm25.py` : recherche par mots-clés
4. `python src/recherche_embeddings.py` : recherche par le sens
5. `python src/rerank.py` : reclassement des 20 meilleurs candidats
6. `python src/entrainer_reranker.py` puis `python src/rerank.py models/reranker-hq` : essai de spécialisation
7. `python src/repondre.py "votre question"` : réponse générée avec sa source (Ollama requis)
8. `python src/evaluer_reponses.py` : évaluation des réponses sur les 50 questions

## Installation

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
python src/ingest_documents.py && python src/decouper.py
ollama pull qwen2.5:7b
streamlit run app.py
```

## Organisation

| Dossier | Contenu |
| --- | --- |
| `src/` | ingestion, découpage, BM25, embeddings, reranker, entraînement, génération, évaluation |
| `data/questions/` | `questions.csv` (jeu de test annoté) et `entrainement.csv` (160 questions synthétiques) |
| `resultats/` | les réponses de chaque modèle, avec la vérification manuelle |
| `app.py` | application Streamlit |

Les PDF officiels ne sont pas publiés ici : `src/ingest_documents.py` les télécharge depuis hydroquebec.com.

## Pistes

- Corriger les 2 réponses de référence incomplètes (q026, q046) et agrandir le jeu de test
- Recherche hybride (BM25 + embeddings) pour les codes courts comme « tarif D »
- Nettoyer l'extraction du texte des PDF (« l e kilowattheure ») avant le découpage
