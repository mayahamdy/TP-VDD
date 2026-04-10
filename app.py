from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Projet VDD - Ciblage Client",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
PRED_PATH = BASE_DIR / "predictions_clients_a_contacter.csv"

COLORS = {
    "bg": "#4B1248",
    "bg_deep": "#2E0A2E",
    "panel": "#5C1A5D",
    "panel_soft": "#6F2B70",
    "violet": "#7B2CBF",
    "violet_soft": "#9D4EDD",
    "violet_light": "#C77DFF",
    "pink": "#FF99C8",
    "pink_soft": "#FFC2DD",
    "text": "#FFD6EB",
    "text_muted": "#F3B6D6",
}


def inject_styles():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top right, rgba(199,125,255,0.24), transparent 28%),
                radial-gradient(circle at top left, rgba(255,153,200,0.16), transparent 32%),
                linear-gradient(180deg, {COLORS["bg"]} 0%, {COLORS["bg_deep"]} 100%);
            color: {COLORS["text"]};
        }}
        .stApp, .stMarkdown, .stText, p, li, label, div {{
            color: {COLORS["text"]};
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {COLORS["panel"]} 0%, {COLORS["bg_deep"]} 100%);
            border-right: 1px solid rgba(255, 214, 235, 0.12);
        }}
        .hero {{
            padding: 1.4rem 1.6rem;
            border-radius: 22px;
            background: linear-gradient(135deg, {COLORS["panel"]} 0%, {COLORS["violet"]} 55%, {COLORS["violet_soft"]} 100%);
            color: {COLORS["text"]};
            box-shadow: 0 18px 40px rgba(46,10,46,0.35);
            margin-bottom: 1rem;
            border: 1px solid rgba(255, 214, 235, 0.15);
        }}
        .hero h1 {{
            margin: 0 0 0.35rem 0;
            font-size: 2.2rem;
            color: {COLORS["pink_soft"]};
        }}
        .hero p {{
            margin: 0.2rem 0;
            font-size: 1rem;
            line-height: 1.6;
            color: {COLORS["text"]};
        }}
        .story-box {{
            padding: 1rem 1.1rem;
            border-radius: 16px;
            background: rgba(111, 43, 112, 0.72);
            border-left: 6px solid {COLORS["pink"]};
            box-shadow: 0 10px 22px rgba(0,0,0,0.18);
            margin-bottom: 0.9rem;
            border: 1px solid rgba(255, 214, 235, 0.12);
        }}
        .metric-box {{
            padding: 1rem;
            border-radius: 16px;
            background: rgba(92, 26, 93, 0.82);
            border: 1px solid rgba(255, 214, 235, 0.12);
            box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        }}
        .section-title {{
            color: {COLORS["pink_soft"]};
            margin-top: 0.2rem;
            margin-bottom: 0.4rem;
        }}
        [data-testid="stDataFrame"], [data-testid="stMetric"], .stPlotlyChart {{
            background: transparent;
        }}
        div[data-baseweb="select"] > div, .stSlider {{
            color: {COLORS["text"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data():
    train = pd.read_csv(DATA_DIR / "train_info.csv")
    prospects = pd.read_csv(DATA_DIR / "clients_a_contacter.csv")
    preds = pd.read_csv(PRED_PATH) if PRED_PATH.exists() else None
    return train, prospects, preds


def normalize_damage_column(df):
    damage_map = {"no": 0, "oui": 1, 0: 0, 1: 1}
    normalized = df["vehicule_endommage"].map(damage_map)
    df["vehicule_endommage"] = normalized.astype("Int64")
    return df


def prepare_data(train_df, prospects_df, preds_df):
    train = train_df.copy()
    prospects = prospects_df.copy()

    train = normalize_damage_column(train)
    prospects = normalize_damage_column(prospects)

    if preds_df is not None:
        prospects = prospects.merge(preds_df, on="id_client", how="left")

    train["tranche_age"] = pd.cut(
        train["age"],
        bins=[18, 25, 35, 45, 55, 65, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
        include_lowest=True,
    )

    prospects["tranche_age"] = pd.cut(
        prospects["age"],
        bins=[18, 25, 35, 45, 55, 65, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
        include_lowest=True,
    )

    return train, prospects


def metric_card(title, value, help_text):
    st.markdown(
        f"""
        <div class="metric-box">
            <div style="font-size:0.92rem; color:{COLORS["pink"]}; font-weight:700;">{title}</div>
            <div style="font-size:2rem; color:{COLORS["pink_soft"]}; font-weight:800; margin:0.15rem 0 0.2rem 0;">{value}</div>
            <div style="font-size:0.92rem; color:{COLORS["text_muted"]}; opacity:0.95;">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_target_distribution(train):
    counts = (
        train["reponse_client"]
        .value_counts()
        .rename_axis("reponse_client")
        .reset_index(name="effectif")
        .sort_values("reponse_client")
    )
    counts["label"] = counts["reponse_client"].map({0: "Réponse négative", 1: "Réponse positive"})
    fig = px.bar(
        counts,
        x="label",
        y="effectif",
        color="label",
        text="effectif",
        color_discrete_map={
            "Réponse négative": COLORS["violet_soft"],
            "Réponse positive": COLORS["pink"],
        },
        title="La première lecture de l'histoire : la cible est déséquilibrée",
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        title_font=dict(color=COLORS["pink_soft"]),
    )
    fig.update_traces(textposition="outside")
    return fig


def plot_response_rate(train, col, title):
    grouped = (
        train.groupby(col, dropna=False, observed=False)["reponse_client"]
        .agg(taux_reponse="mean", effectif="size")
        .reset_index()
        .sort_values("taux_reponse", ascending=False)
    )
    fig = px.bar(
        grouped,
        x=col,
        y="taux_reponse",
        color="taux_reponse",
        text="effectif",
        color_continuous_scale=[COLORS["violet_light"], COLORS["violet_soft"], COLORS["pink"]],
        title=title,
        hover_data={"effectif": True, "taux_reponse": ":.2%"},
    )
    fig.update_traces(texttemplate="n=%{text}", textposition="outside")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        title_font=dict(color=COLORS["pink_soft"]),
    )
    return fig


def plot_numeric_distribution(train, variable):
    fig = px.histogram(
        train,
        x=variable,
        color="reponse_client",
        marginal="box",
        nbins=45,
        opacity=0.7,
        barmode="overlay",
        color_discrete_map={0: COLORS["violet_soft"], 1: COLORS["pink"]},
        title=f"Comment la variable {variable} se répartit selon la réponse client",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Réponse",
        font=dict(color=COLORS["text"]),
        title_font=dict(color=COLORS["pink_soft"]),
    )
    return fig


def plot_priority_segments(prospects):
    if "segment_contact" not in prospects.columns:
        return None
    counts = prospects["segment_contact"].value_counts().rename_axis("segment").reset_index(name="effectif")
    color_map = {
        "faible_priorite": COLORS["violet_soft"],
        "priorite_intermediaire": COLORS["violet_light"],
        "haute_priorite": COLORS["pink"],
    }
    fig = px.pie(
        counts,
        names="segment",
        values="effectif",
        hole=0.45,
        color="segment",
        color_discrete_map=color_map,
        title="Répartition des segments de contact proposés par le modèle",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        title_font=dict(color=COLORS["pink_soft"]),
    )
    return fig


def plot_probability_histogram(prospects):
    if "proba_reponse" not in prospects.columns:
        return None
    fig = px.histogram(
        prospects,
        x="proba_reponse",
        color="segment_contact" if "segment_contact" in prospects.columns else None,
        nbins=40,
        opacity=0.85,
        color_discrete_map={
            "faible_priorite": COLORS["violet_soft"],
            "priorite_intermediaire": COLORS["violet_light"],
            "haute_priorite": COLORS["pink"],
        },
        title="Distribution des probabilités prédites : où se trouvent les meilleurs profils ?",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Probabilité de réponse positive",
        font=dict(color=COLORS["text"]),
        title_font=dict(color=COLORS["pink_soft"]),
    )
    return fig


def top_profiles_table(prospects):
    if "proba_reponse" not in prospects.columns:
        return None
    cols = [
        "id_client",
        "genre",
        "age",
        "age_vehicule",
        "vehicule_endommage",
        "prime_annuelle",
        "anciennete",
        "proba_reponse",
        "segment_contact",
    ]
    available = [c for c in cols if c in prospects.columns]
    table = prospects[available].sort_values("proba_reponse", ascending=False).head(20).copy()
    if "proba_reponse" in table.columns:
        table["proba_reponse"] = table["proba_reponse"].map(lambda x: f"{x:.1%}")
    return table


inject_styles()
train_df, prospects_df, preds_df = load_data()
train_df, prospects_df = prepare_data(train_df, prospects_df, preds_df)

response_rate = train_df["reponse_client"].mean()
missing_total = int(train_df.isna().sum().sum())
avg_premium = train_df["prime_annuelle"].mean()

st.markdown(
    f"""
    <div class="hero">
        <h1>Ciblage client : raconter l'histoire derrière les données</h1>
        <p>Cette application suit un fil simple : d'abord comprendre qui sont les clients, ensuite repérer les signaux liés à la réponse positive, puis transformer cette lecture en une liste de contacts priorisés.</p>
        <p>Le ton est volontairement narratif : chaque graphique répond à une question métier claire, avec des mesures visibles et des repères faciles à interpréter.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Lecture guidée")
    st.write("1. Vue d'ensemble")
    st.write("2. Signaux de réponse")
    st.write("3. Lecture des profils")
    st.write("4. Ciblage opérationnel")
    st.divider()
    numeric_focus = st.selectbox(
        "Variable quantitative à explorer",
        ["age", "prime_annuelle", "anciennete"],
    )
    top_n = st.slider("Nombre de clients prioritaires à afficher", 10, 100, 20, 10)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Le décor", "Ce qui influence la réponse", "Le modèle raconte quoi ?", "Passage à l'action"]
)

with tab1:
    st.markdown('<h2 class="section-title">Le décor</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="story-box">
        Avant de prédire quoi que ce soit, il faut comprendre le terrain. Ici, on dispose d'un grand historique de clients déjà observés et d'un second fichier de prospects à classer. Cette première section donne le contexte général : taille des données, qualité et composition globale.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Clients d'entraînement", f"{len(train_df):,}".replace(",", " "), "Base utilisée pour apprendre le comportement des clients.")
    with c2:
        metric_card("Clients à scorer", f"{len(prospects_df):,}".replace(",", " "), "Population sur laquelle on applique le ciblage final.")
    with c3:
        metric_card("Taux de réponse positive", f"{response_rate:.1%}", "Proportion de clients positifs dans le fichier d'entraînement.")
    with c4:
        metric_card("Valeurs manquantes", str(missing_total), "Le jeu est très propre sur ce point.")

    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(plot_target_distribution(train_df), width="stretch")
    with right:
        st.markdown(
            """
            <div class="story-box">
            <b>Lecture rapide :</b> la classe positive est minoritaire. Cela veut dire qu'un modèle naïf pourrait facilement dire "non" presque tout le temps. Il faut donc lire les performances avec soin et s'intéresser à la capacité du modèle à retrouver les clients réellement intéressés.
            </div>
            """,
            unsafe_allow_html=True,
        )
        quality = pd.DataFrame(
            {
                "Type": train_df.dtypes.astype(str),
                "Manquants": train_df.isna().sum(),
                "Valeurs uniques": train_df.nunique(),
            }
        )
        st.dataframe(quality, width="stretch")

    st.markdown("### Un aperçu concret des données")
    st.dataframe(train_df.head(12), width="stretch")

with tab2:
    st.markdown('<h2 class="section-title">Ce qui influence la réponse</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="story-box">
        Cette partie cherche les signaux les plus parlants. L'idée est simple : comparer les groupes de clients et voir où le taux de réponse monte vraiment. On ne cherche pas encore à conclure causalement, mais à identifier des profils prometteurs.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            plot_response_rate(train_df, "vehicule_endommage", "Les clients avec véhicule endommagé répondent-ils plus souvent ?"),
            width="stretch",
        )
    with col_b:
        st.plotly_chart(
            plot_response_rate(train_df, "age_vehicule", "L'âge du véhicule joue-t-il sur la réponse ?"),
            width="stretch",
        )

    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(
            plot_response_rate(train_df, "genre", "Le taux de réponse varie-t-il selon le genre ?"),
            width="stretch",
        )
    with col_d:
        st.plotly_chart(
            plot_response_rate(train_df, "tranche_age", "La tranche d'âge fait-elle apparaître des groupes plus réactifs ?"),
            width="stretch",
        )

    st.plotly_chart(plot_numeric_distribution(train_df, numeric_focus), width="stretch")

    rate_table = (
        train_df.groupby(["vehicule_endommage", "age_vehicule"], observed=False)["reponse_client"]
        .mean()
        .reset_index()
        .sort_values("reponse_client", ascending=False)
        .rename(columns={"reponse_client": "taux_reponse"})
    )
    rate_table["taux_reponse"] = rate_table["taux_reponse"].map(lambda x: f"{x:.1%}")

    st.markdown(
        """
        <div class="story-box">
        <b>Point de lecture métier :</b> les variables liées à l'état et à l'ancienneté du véhicule ressortent fortement. Ce sont de bons candidats pour expliquer ou au moins anticiper la probabilité de souscription.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(rate_table, width="stretch")

with tab3:
    st.markdown('<h2 class="section-title">Le modèle raconte quoi ?</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="story-box">
        Une fois le modèle entraîné, l'information la plus intéressante n'est pas seulement la classe 0 ou 1, mais surtout la <b>probabilité</b> attribuée à chaque prospect. C'est cette note qui permet ensuite d'organiser la prise de contact.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if preds_df is None or "proba_reponse" not in prospects_df.columns:
        st.warning("Le fichier de prédictions n'a pas été trouvé. Lance d'abord le notebook pour générer `predictions_clients_a_contacter.csv`.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card(
                "Probabilité moyenne",
                f"{prospects_df['proba_reponse'].mean():.1%}",
                "Niveau moyen de réponse positive estimé sur les prospects.",
            )
        with col2:
            metric_card(
                "Prospects haute priorité",
                f"{(prospects_df['segment_contact'] == 'haute_priorite').sum():,}".replace(",", " "),
                "Clients que le modèle place dans la zone la plus prometteuse.",
            )
        with col3:
            metric_card(
                "Prime annuelle moyenne",
                f"{avg_premium:,.0f}".replace(",", " "),
                "Repère utile pour relier ciblage et valeur potentielle.",
            )

        left, right = st.columns(2)
        with left:
            hist_fig = plot_probability_histogram(prospects_df)
            if hist_fig is not None:
                st.plotly_chart(hist_fig, width="stretch")
        with right:
            pie_fig = plot_priority_segments(prospects_df)
            if pie_fig is not None:
                st.plotly_chart(pie_fig, width="stretch")

        segment_summary = (
            prospects_df.groupby("segment_contact", observed=False)
            .agg(
                nb_clients=("id_client", "size"),
                age_moyen=("age", "mean"),
                prime_moyenne=("prime_annuelle", "mean"),
                anciennete_moyenne=("anciennete", "mean"),
            )
            .reset_index()
            .sort_values("nb_clients", ascending=False)
        )
        st.markdown("### Lecture détaillée des segments")
        st.dataframe(segment_summary, width="stretch")

with tab4:
    st.markdown("<h2 class=\"section-title\">Passage à l'action</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="story-box">
        La dernière étape traduit l'analyse en action. L'objectif n'est pas seulement de produire un score, mais de donner une liste claire de clients à cibler et une justification simple à présenter dans un rendu ou à l'oral.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if preds_df is None or "proba_reponse" not in prospects_df.columns:
        st.info("Aucune table de scoring disponible pour le moment.")
    else:
        top_table = (
            prospects_df.sort_values("proba_reponse", ascending=False)
            .head(top_n)
            .copy()
        )

        st.markdown("### Les prospects les plus prometteurs")
        display_table = top_profiles_table(prospects_df)
        if display_table is not None:
            st.dataframe(display_table.head(top_n), width="stretch")

        profile_fig = px.scatter(
            top_table,
            x="age",
            y="prime_annuelle",
            size="anciennete",
            color="segment_contact",
            hover_data=["id_client", "age_vehicule", "vehicule_endommage"],
            color_discrete_map={
                "faible_priorite": COLORS["violet_soft"],
                "priorite_intermediaire": COLORS["violet_light"],
                "haute_priorite": COLORS["pink"],
            },
            title="Carte rapide des meilleurs prospects : âge, prime et ancienneté",
        )
        profile_fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
            title_font=dict(color=COLORS["pink_soft"]),
        )
        st.plotly_chart(profile_fig, width="stretch")

        st.markdown(
            """
            <div class="story-box">
            <b>Recommandation finale :</b> commencer par les profils en <b>haute priorité</b>, puis élargir si besoin à la <b>priorité intermédiaire</b>. Cette logique permet d'éviter un ciblage trop large et de concentrer l'effort commercial sur les profils les plus crédibles.
            </div>
            """,
            unsafe_allow_html=True,
        )

        csv_data = prospects_df.sort_values("proba_reponse", ascending=False).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Télécharger la table de scoring complète",
            data=csv_data,
            file_name="clients_scores_dashboard.csv",
            mime="text/csv",
        )
