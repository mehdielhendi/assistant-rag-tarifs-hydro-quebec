"""Application web : poser des questions sur les tarifs d'Hydro-Québec et voir l'évaluation du système.

Lancer en local :  streamlit run app.py
"""

import re
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ingest_documents import BASE, DOCUMENTS  # noqa: E402
from recherche_embeddings import PASSAGES  # noqa: E402

st.set_page_config(page_title="Tarifs Hydro-Québec · Assistant RAG", page_icon="⚡", layout="wide")

PDF = {Path(nom).stem: BASE + fichier for nom, fichier in DOCUMENTS.items()}
NOMS = {"tarifs-2026": "Tarifs d'électricité 2026", "conditions-service-2026": "Conditions de service 2026",
        "grille-tarifaire-2026": "Grille tarifaire 2026"}
EXEMPLES = [
    ("💡", "Combien coûte le kWh au tarif D ?"),
    ("❄️", "Est-ce qu'Hydro-Québec peut me couper l'électricité en plein hiver si je ne paie pas ?"),
    ("📲", "Avec Flex D, comment je suis prévenu d'un événement de pointe ?"),
    ("☀️", "J'ai des panneaux solaires, quelle option me permet d'utiliser mes surplus ?"),
    ("🧾", "J'ai combien de temps pour payer ma facture ?"),
    ("🚫", "Quel est le prix de l'électricité en Ontario ?"),
]

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1rem;
       background: linear-gradient(120deg, rgba(31,111,235,.18), rgba(245,183,0,.12));
       border: 1px solid rgba(128,128,128,.25);}
.hero h1 {font-size: 2rem; margin: 0 0 .3rem 0; padding: 0;}
.hero p {margin: 0; opacity: .85;}
.chips {margin-top: .7rem;}
.chip {display: inline-block; padding: .15rem .6rem; margin: .15rem .3rem 0 0; border-radius: 999px;
       font-size: .8rem; border: 1px solid rgba(128,128,128,.35);}
.badge {display: inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .78rem;
        font-weight: 600; margin-bottom: .4rem;}
.ok {background: rgba(46,160,67,.18); color: #2ea043;}
.non {background: rgba(219,154,4,.18); color: #d29922;}
.meta {font-size: .8rem; opacity: .7;}
</style>
""", unsafe_allow_html=True)


# ---------- moteur ----------

@st.cache_resource(show_spinner="Premier lancement : téléchargement des documents et chargement des modèles…")
def assistant():
    if not PASSAGES.exists():
        import decouper
        import ingest_documents
        ingest_documents.telecharger()
        decouper.main()
    from repondre import Assistant
    return Assistant()


def proteger(texte: str) -> str:
    """Évite que Streamlit lise « 15 $ … 46 $ » comme une formule mathématique."""
    return texte.replace("$", r"\$")


def lien_pdf(document: str, page: int) -> str:
    return f"{PDF.get(document, '')}#page={page}"


def mettre_en_forme_reponse(texte: str) -> str:
    """Rend la source cliquable : (Source : tarifs-2026, page 34) -> lien vers la page du PDF officiel."""
    def lien(m):
        doc, page = m.group(1).strip(), int(m.group(2))
        return f"[📄 {NOMS.get(doc, doc)}, p. {page}]({lien_pdf(doc, page)})" if doc in PDF else m.group(0)
    return re.sub(r"\(?Source\s*:\s*([\w-]+),\s*page\s*(\d+)\)?\.?", lien, proteger(texte))


def embellir_passage(texte: str) -> str:
    texte = re.sub(r"\b([ldLD]) e\b", r"\1e", texte)          # « l e kilowattheure » -> « le kilowattheure »
    texte = re.sub(r"(\d[\d  ]*(?:,\d+)?\s?(?:¢|\$|%|kWh|kW\b|kilowattheures?|kilowatts?|jours|mois|heures))",
                   r"**\1**", texte)
    return proteger(texte)


def est_refus(texte: str) -> bool:
    return "ne trouve pas cette information" in texte.lower()


# ---------- affichage d'une réponse ----------

def afficher_sources(tour: dict) -> None:
    for n, p in enumerate(tour["extraits"], start=1):
        titre = (f"Extrait {n} · {NOMS.get(p['document'], p['document'])}, page {p['page']} · "
                 f"pertinence {p['pertinence']:.0%}")
        with st.expander(titre):
            st.markdown(embellir_passage(p["texte"]))
            st.link_button("Ouvrir la page dans le PDF officiel ↗", lien_pdf(p["document"], p["page"]))
    if st.session_state.get("detail"):
        df = pd.DataFrame(tour["classement"])
        df["mouvement"] = df["rang_embeddings"] - df["rang_reranker"]
        st.caption("Comment le reranker a reclassé les 20 candidats trouvés par les embeddings")
        st.dataframe(
            df[["rang_reranker", "rang_embeddings", "mouvement", "document", "page", "pertinence", "similarite"]],
            hide_index=True, width="stretch", height=260,
            column_config={
                "rang_reranker": "Rang final", "rang_embeddings": "Rang embeddings",
                "mouvement": st.column_config.NumberColumn("Gain", format="%+d"),
                "pertinence": st.column_config.ProgressColumn("Pertinence (reranker)", min_value=0, max_value=1),
                "similarite": st.column_config.NumberColumn("Similarité cosinus", format="%.3f"),
            })


def afficher_tour(tour: dict) -> None:
    with st.chat_message("user", avatar="🙋"):
        st.markdown(proteger(tour["question"]))
    with st.chat_message("assistant", avatar="⚡"):
        badge = ('<span class="badge non">Pas dans les documents</span>' if est_refus(tour["reponse"])
                 else '<span class="badge ok">Réponse trouvée dans les documents</span>')
        st.markdown(badge, unsafe_allow_html=True)
        st.markdown(mettre_en_forme_reponse(tour["reponse"]))
        st.markdown(f'<div class="meta">Recherche {tour["t_recherche"]:.1f} s · rédaction '
                    f'{tour["t_generation"]:.1f} s · {tour["modele"]}</div>', unsafe_allow_html=True)
        afficher_sources(tour)


def traiter(question: str) -> None:
    from repondre import appeler_llm_flux, construire_messages, nom_modele

    with st.chat_message("user", avatar="🙋"):
        st.markdown(proteger(question))
    with st.chat_message("assistant", avatar="⚡"):
        try:
            with st.spinner("Recherche des passages pertinents…"):
                debut = time.perf_counter()
                classement = assistant().classer(question)
                extraits = classement[:st.session_state.get("n_passages", 3)]
                t_recherche = time.perf_counter() - debut
            debut = time.perf_counter()
            zone = st.empty()
            with zone.container():
                reponse = st.write_stream(proteger(m) for m in
                                          appeler_llm_flux(construire_messages(question, extraits)))
            zone.empty()
            t_generation = time.perf_counter() - debut
        except Exception as erreur:                     # serveur du modèle arrêté, quota atteint, etc.
            st.error(f"Le modèle de langue n'a pas pu répondre : {erreur}")
            return
    reponse = reponse.replace(r"\$", "$")
    st.session_state.historique.append({
        "question": question, "reponse": reponse, "extraits": extraits, "classement": classement,
        "t_recherche": t_recherche, "t_generation": t_generation, "modele": nom_modele()})
    st.rerun()


# ---------- onglets ----------

def onglet_assistant() -> None:
    historique = st.session_state.historique
    if not historique:
        st.markdown("#### Essayez une question")
        colonnes = st.columns(3)
        for k, (icone, texte) in enumerate(EXEMPLES):
            if colonnes[k % 3].button(f"{icone} {texte}", key=f"ex{k}", width="stretch"):
                st.session_state.en_attente = texte
                st.rerun()
        st.caption("La dernière question n'a pas de réponse dans les documents : l'assistant doit le dire "
                   "au lieu d'inventer.")
    for tour in historique:
        afficher_tour(tour)

    question = st.chat_input("Posez votre question sur les tarifs ou les conditions de service…")
    question = question or st.session_state.pop("en_attente", None)
    if question and question.strip():
        traiter(question.strip())


def onglet_evaluation() -> None:
    st.markdown("Jeu de test : **50 questions écrites à la main** (44 avec une réponse dans les documents, "
                "6 hors corpus), chacune vérifiée dans les PDF. Aucune n'a servi à l'entraînement.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Réponses correctes", "84 %", "37 / 44", delta_color="off", border=True)
    c2.metric("Inventions hors corpus", "0 / 6", "refus correct 100 %", delta_color="off", border=True)
    c3.metric("Refus à tort", "9 %", "-27 pts vs 3B", delta_color="inverse", border=True)
    c4.metric("Bonne page dans le top 3", "84 %", "+34 pts vs BM25", border=True)

    st.subheader("1 · Trouver la bonne page")
    recherche = pd.DataFrame({
        "Méthode": ["BM25 (mots-clés)"] * 4 + ["Embeddings e5-small"] * 4 + ["Embeddings + reranker"] * 4,
        "k": ["Top 1", "Top 3", "Top 5", "Top 10"] * 3,
        "Taux": [.23, .50, .66, .73, .455, .727, .841, .909, .636, .841, .909, .932],
    })
    ordre = list(dict.fromkeys(recherche["Méthode"]))
    graphique = alt.Chart(recherche).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("k:N", title=None, sort=["Top 1", "Top 3", "Top 5", "Top 10"]),
        xOffset=alt.XOffset("Méthode:N", sort=ordre),
        y=alt.Y("Taux:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title="Bonne page trouvée"),
        color=alt.Color("Méthode:N", sort=ordre, scale=alt.Scale(range=["#9aa4b2", "#5b8def", "#1f6feb"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Méthode", "k", alt.Tooltip("Taux:Q", format=".1%")],
    ).properties(height=320, width="container")
    st.altair_chart(graphique)
    st.caption("Essai de spécialisation du reranker (PyTorch, 160 questions synthétiques sur d'autres pages) : "
               "top 1 de 64 % à 57 %. Résultat négatif documenté ; le reranker pré-entraîné est conservé.")

    st.subheader("2 · Répondre sans inventer")
    st.dataframe(pd.DataFrame({
        "Modèle": ["qwen2.5:3b", "qwen2.5:7b"],
        "Refus correct hors corpus": ["6/6", "6/6"],
        "Refus à tort": ["16/44 (36 %)", "4/44 (9 %)"],
        "Bonne page citée": ["59 %", "73 %"],
        "Correctes (vérification manuelle)": ["—", "37/44 (84 %)"],
    }), hide_index=True, width="stretch")
    st.caption("Mesures faites en local avec Ollama. La démo en ligne utilise un autre modèle (API Groq).")

    detail = Path("resultats/reponses_qwen2.5-7b.csv")
    if detail.exists():
        st.subheader("3 · Les 50 réponses, une par une")
        df = pd.read_csv(detail)
        filtre = st.segmented_control("Afficher", ["Toutes", "Correctes", "Erreurs", "Hors corpus"],
                                      default="Toutes")
        if filtre == "Correctes":
            df = df[df["correct_manuel"] == 1]
        elif filtre == "Erreurs":
            df = df[df["correct_manuel"] == 0]
        elif filtre == "Hors corpus":
            df = df[df["reponse_attendue"] == "NON_COUVERT"]
        df["Verdict"] = df["correct_manuel"].map({1: "✅", 0: "❌"}).fillna("🚫 hors corpus")
        st.dataframe(df[["id", "Verdict", "question", "reponse_attendue", "reponse_generee", "note"]],
                     hide_index=True, width="stretch", height=420,
                     column_config={"reponse_attendue": "Réponse attendue",
                                    "reponse_generee": st.column_config.TextColumn("Réponse générée", width="large"),
                                    "question": st.column_config.TextColumn("Question", width="medium")})


def onglet_methode() -> None:
    st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#1f6feb22", color="#1f6feb", fontname="Helvetica", fontsize=11];
  edge [color="#8b949e"];
  Q [label="Question"];
  E [label="Embeddings\\nmultilingual-e5-small\\n838 passages → 20"];
  R [label="Reranker\\ncross-encoder mMiniLM\\n20 → 3"];
  L [label="LLM\\nconsigne stricte\\n+ citation"];
  A [label="Réponse sourcée\\nou refus", fillcolor="#f5b70033", color="#d29922"];
  Q -> E -> R -> L -> A;
}""")
    st.markdown("""
**Corpus.** Trois PDF officiels d'Hydro-Québec, découpés en 838 passages de 180 mots (chevauchement de 40),
chacun rattaché à son document et à sa page.

**Recherche.** Les embeddings comparent le sens de la question et des passages et gardent les 20 plus proches.
Le *cross-encoder* lit ensuite chaque paire (question, passage) en entier et garde les 3 meilleurs.

**Génération.** Le LLM reçoit les 3 passages avec quatre règles : répondre seulement avec les extraits,
recopier les chiffres exactement, citer la page, et refuser si la réponse n'y est pas.

**Évaluation.** Rang de la bonne page, refus à tort, réponses inventées sur des questions hors corpus,
puis vérification manuelle de chaque réponse.

**Ce qui est fait à la main.** BM25 implémenté sans bibliothèque, boucle d'entraînement PyTorch du reranker,
jeu de test annoté, scripts d'évaluation reproductibles.
""")
    st.markdown("**Documents sources**")
    for doc, url in PDF.items():
        st.markdown(f"- [{NOMS.get(doc, doc)}]({url})")


# ---------- page ----------

st.session_state.setdefault("historique", [])

with st.sidebar:
    st.markdown("### ⚡ Assistant tarifs")
    st.caption("Projet de El Hendi Mohamed Mehdi · maîtrise en IA, Université de Sherbrooke")
    st.divider()
    st.markdown("**Réglages**")
    st.slider("Passages donnés au modèle", 1, 5, 3, key="n_passages",
              help="L'évaluation a été faite avec 3 passages.")
    st.toggle("Montrer le reclassement des 20 candidats", key="detail",
              help="Affiche le rang de chaque passage avant et après le reranker.")
    if st.button("🗑️ Effacer la conversation", width="stretch", disabled=not st.session_state.historique):
        st.session_state.historique = []
        st.rerun()
    st.divider()
    from repondre import nom_modele
    st.markdown(f"**Modèle** : `{nom_modele()}`  \n**Recherche** : e5-small + mMiniLM  \n"
                "**Corpus** : 3 PDF, 838 passages")
    st.caption("Non affilié à Hydro-Québec. Vérifiez toujours dans les documents officiels.")

st.markdown("""
<div class="hero">
  <h1>⚡ Questions sur les tarifs d'Hydro-Québec</h1>
  <p>Posez une question comme un client : l'assistant répond <b>uniquement</b> à partir des documents officiels,
  cite la page, et dit quand il ne sait pas.</p>
  <div class="chips"><span class="chip">RAG</span><span class="chip">Embeddings + reranker</span>
  <span class="chip">LLM local ou API</span><span class="chip">Évalué sur 50 questions</span></div>
</div>
""", unsafe_allow_html=True)

t1, t2, t3 = st.tabs(["💬 Assistant", "📊 Évaluation", "🛠️ Méthode"])
with t1:
    onglet_assistant()
with t2:
    onglet_evaluation()
with t3:
    onglet_methode()
