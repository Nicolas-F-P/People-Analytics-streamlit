import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from pandas.api.types import is_numeric_dtype



# Configuration de la page


st.set_page_config(
    page_title="People Analytics - Attrition RH",
    page_icon="👥",
    layout="wide"
)



# Chargement des fichiers


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "employee_attrition_clean.csv"
MODEL_PATH = BASE_DIR / "models" / "attrition_final_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "model_features.pkl"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_features():
    return joblib.load(FEATURES_PATH)


df = load_data()
model = load_model()
model_features = load_features()



# Titre de l'application


st.title("👥 People Analytics — Prédiction du turnover RH")

st.markdown(
    """
    Cette application analyse les facteurs associés au départ des employés et permet de simuler
    le risque d'attrition d'un profil salarié.

    Le modèle utilisé est un modèle de classification entraîné sur un dataset RH.
    Il ne remplace pas une décision humaine, mais sert d'outil d'aide à l'analyse.
    """
)



# Sidebar


st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choisir une page",
    [
        "Vue d'ensemble",
        "Analyse des facteurs",
        "Simulateur de risque",
        "Recommandations RH"
    ]
)



# Page 1 — Vue d'ensemble


if page == "Vue d'ensemble":
    st.header("Vue d'ensemble RH")

    total_employees = df.shape[0]
    employees_left = int(df["Attrition_Flag"].sum())
    employees_stayed = total_employees - employees_left
    attrition_rate = df["Attrition_Flag"].mean() * 100
    average_age = df["Age"].mean()
    average_income = df["MonthlyIncome"].mean()
    average_years_company = df["YearsAtCompany"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Employés", f"{total_employees}")
    col2.metric("Employés partis", f"{employees_left}")
    col3.metric("Taux d'attrition", f"{attrition_rate:.2f}%")
    col4.metric("Ancienneté moyenne", f"{average_years_company:.1f} ans")

    col5, col6, col7 = st.columns(3)

    col5.metric("Employés restés", f"{employees_stayed}")
    col6.metric("Âge moyen", f"{average_age:.1f} ans")
    col7.metric("Salaire mensuel moyen", f"{average_income:.0f}")

    st.subheader("Répartition de l'attrition")

    attrition_counts = df["Attrition"].value_counts()
    st.bar_chart(attrition_counts)

    st.subheader("Taux d'attrition par département")

    attrition_by_department = (
        df.groupby("Department")["Attrition_Flag"]
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )

    st.bar_chart(attrition_by_department)



# Page 2 — Analyse des facteurs


elif page == "Analyse des facteurs":
    st.header("Analyse des facteurs associés à l'attrition")

    st.markdown(
        """
        Cette page permet d'observer les taux d'attrition selon plusieurs variables importantes :
        heures supplémentaires, poste, satisfaction, ancienneté et équilibre vie professionnelle / personnelle.
        """
    )

    selected_factor = st.selectbox(
        "Choisir un facteur à analyser",
        [
            "OverTime",
            "JobRole",
            "JobSatisfaction",
            "WorkLifeBalance",
            "YearsAtCompany",
            "Age",
            "Department"
        ]
    )

    temp_df = df.copy()

    if selected_factor == "YearsAtCompany":
        temp_df["YearsAtCompany_Group"] = pd.cut(
            temp_df["YearsAtCompany"],
            bins=[-1, 1, 3, 5, 10, 40],
            labels=["0-1 an", "2-3 ans", "4-5 ans", "6-10 ans", "10+ ans"]
        )
        factor_column = "YearsAtCompany_Group"

    elif selected_factor == "Age":
        temp_df["Age_Group"] = pd.cut(
            temp_df["Age"],
            bins=[17, 25, 35, 45, 55, 65],
            labels=["18-25", "26-35", "36-45", "46-55", "56+"]
        )
        factor_column = "Age_Group"

    else:
        factor_column = selected_factor

    factor_analysis = (
        temp_df.groupby(factor_column)["Attrition_Flag"]
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )

    st.subheader(f"Taux d'attrition selon : {selected_factor}")
    st.bar_chart(factor_analysis)

    st.dataframe(factor_analysis.reset_index().rename(columns={"Attrition_Flag": "Taux d'attrition (%)"}))



# Page 3 — Simulateur de risque


elif page == "Simulateur de risque":
    st.header("Simulateur de risque de départ")

    st.markdown(
        """
        Remplis les informations principales d'un profil salarié.
        L'application estime ensuite un score de risque d'attrition.
        """
    )

    # Valeurs de base : on part d'un profil moyen
    employee_profile = {}

    for feature in model_features:
        if feature in df.columns:
            if is_numeric_dtype(df[feature]):
                employee_profile[feature] = df[feature].median()
            else:
                employee_profile[feature] = df[feature].mode()[0]
        else:
                employee_profile[feature] = 0

    col1, col2 = st.columns(2)

    with col1:
        employee_profile["Age"] = st.slider(
            "Âge",
            int(df["Age"].min()),
            int(df["Age"].max()),
            int(df["Age"].median())
        )

        employee_profile["Department"] = st.selectbox(
            "Département",
            sorted(df["Department"].unique())
        )

        employee_profile["JobRole"] = st.selectbox(
            "Poste",
            sorted(df["JobRole"].unique())
        )

        employee_profile["MonthlyIncome"] = st.slider(
            "Salaire mensuel",
            int(df["MonthlyIncome"].min()),
            int(df["MonthlyIncome"].max()),
            int(df["MonthlyIncome"].median())
        )

        employee_profile["DistanceFromHome"] = st.slider(
            "Distance domicile-travail",
            int(df["DistanceFromHome"].min()),
            int(df["DistanceFromHome"].max()),
            int(df["DistanceFromHome"].median())
        )

    with col2:
        employee_profile["OverTime"] = st.selectbox(
            "Heures supplémentaires",
            sorted(df["OverTime"].unique())
        )

        employee_profile["BusinessTravel"] = st.selectbox(
            "Déplacements professionnels",
            sorted(df["BusinessTravel"].unique())
        )

        employee_profile["JobSatisfaction"] = st.slider(
            "Satisfaction au travail",
            int(df["JobSatisfaction"].min()),
            int(df["JobSatisfaction"].max()),
            int(df["JobSatisfaction"].median())
        )

        employee_profile["EnvironmentSatisfaction"] = st.slider(
            "Satisfaction environnement",
            int(df["EnvironmentSatisfaction"].min()),
            int(df["EnvironmentSatisfaction"].max()),
            int(df["EnvironmentSatisfaction"].median())
        )

        employee_profile["WorkLifeBalance"] = st.slider(
            "Équilibre vie pro / perso",
            int(df["WorkLifeBalance"].min()),
            int(df["WorkLifeBalance"].max()),
            int(df["WorkLifeBalance"].median())
        )

        employee_profile["YearsAtCompany"] = st.slider(
            "Ancienneté dans l'entreprise",
            int(df["YearsAtCompany"].min()),
            int(df["YearsAtCompany"].max()),
            int(df["YearsAtCompany"].median())
        )

        employee_profile["TotalWorkingYears"] = st.slider(
            "Expérience professionnelle totale",
            int(df["TotalWorkingYears"].min()),
            int(df["TotalWorkingYears"].max()),
            int(df["TotalWorkingYears"].median())
        )

    input_df = pd.DataFrame([employee_profile])
    input_df = input_df[model_features]

    if st.button("Estimer le risque"):
        risk_probability = model.predict_proba(input_df)[0][1]
        risk_percent = risk_probability * 100

        st.subheader("Résultat de la simulation")

        if risk_percent >= 60:
            risk_level = "Élevé"
            st.error(f"Risque estimé : {risk_percent:.1f}% — Niveau : {risk_level}")
        elif risk_percent >= 35:
            risk_level = "Modéré"
            st.warning(f"Risque estimé : {risk_percent:.1f}% — Niveau : {risk_level}")
        else:
            risk_level = "Faible"
            st.success(f"Risque estimé : {risk_percent:.1f}% — Niveau : {risk_level}")

        st.markdown("### Points de vigilance détectés")

        alerts = []

        if employee_profile["OverTime"] == "Yes":
            alerts.append("L'employé effectue des heures supplémentaires.")

        if employee_profile["JobSatisfaction"] <= 2:
            alerts.append("La satisfaction au travail est faible ou moyenne basse.")

        if employee_profile["WorkLifeBalance"] <= 2:
            alerts.append("L'équilibre vie professionnelle / personnelle est faible.")

        if employee_profile["YearsAtCompany"] <= 1:
            alerts.append("L'employé a une faible ancienneté dans l'entreprise.")

        if employee_profile["BusinessTravel"] == "Travel_Frequently":
            alerts.append("L'employé voyage fréquemment pour le travail.")

        if employee_profile["DistanceFromHome"] >= 15:
            alerts.append("La distance domicile-travail est élevée.")

        if alerts:
            for alert in alerts:
                st.write(f"- {alert}")
        else:
            st.write("Aucun signal de vigilance majeur détecté sur les variables principales.")



# Page 4 — Recommandations RH


elif page == "Recommandations RH":
    st.header("Recommandations RH")

    st.markdown(
        """
        Les analyses réalisées montrent plusieurs axes d'action possibles pour réduire le turnover.
        Ces recommandations sont basées sur les tendances observées dans le dataset et sur les résultats du modèle.
        """
    )

    recommendations = pd.DataFrame(
        {
            "Signal observé": [
                "Heures supplémentaires fréquentes",
                "Forte attrition chez les nouveaux employés",
                "Faible satisfaction au travail",
                "Faible équilibre vie pro / vie perso",
                "Déplacements professionnels fréquents",
                "Postes commerciaux et techniques plus exposés"
            ],
            "Risque potentiel": [
                "Fatigue, surcharge, désengagement",
                "Onboarding insuffisant ou mauvaise intégration",
                "Perte de motivation",
                "Risque de burn-out ou frustration",
                "Contraintes personnelles et fatigue",
                "Pression métier, objectifs ou conditions de travail"
            ],
            "Action recommandée": [
                "Suivre la charge de travail et limiter les heures supplémentaires répétées",
                "Renforcer le parcours d'intégration sur les 12 premiers mois",
                "Mettre en place des enquêtes internes et entretiens ciblés",
                "Favoriser la flexibilité et mieux répartir la charge",
                "Adapter la fréquence des déplacements si possible",
                "Analyser les conditions spécifiques de ces postes"
            ]
        }
    )

    st.dataframe(recommendations, use_container_width=True)

    st.markdown(
        """
        ### Limites de l'analyse

        Le modèle identifie des associations statistiques, mais ne prouve pas de causalité.
        Dans un contexte réel, ces résultats devraient être complétés par des entretiens RH,
        des données qualitatives et une analyse interne plus détaillée.
        """
    )