# People Analytics - Analyse et prédiction du turnover RH

## 1. Contexte du projet

Ce projet analyse un dataset RH afin d'identifier les facteurs associés au départ des employés et de construire un outil simple d'aide à la décision.

Le sujet choisi est volontairement différent des autres projets du portfolio : ici, l'objectif n'est pas seulement de produire un dashboard, mais de construire une démarche complète autour de Python, des statistiques, du machine learning et d'une application interactive Streamlit.

**Problématique métier :**

> Comment identifier les profils d'employés les plus exposés au risque de départ et aider une direction RH à prioriser ses actions de rétention ?

---

## 2. Objectifs

Le projet poursuit plusieurs objectifs :

- comprendre la structure d'un dataset RH ;
- nettoyer et préparer les données ;
- analyser les principaux facteurs associés à l'attrition ;
- vérifier certaines relations avec des tests statistiques ;
- construire un modèle prédictif simple ;
- interpréter les variables importantes ;
- créer une application Streamlit permettant de simuler un risque de départ ;
- formuler des recommandations RH compréhensibles par un public métier.

---

## 3. Dataset utilisé

Le dataset utilisé est **IBM HR Analytics Employee Attrition & Performance**, disponible sur Kaggle :

https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Il contient des informations relatives aux employés : âge, département, poste, satisfaction, salaire mensuel, ancienneté, heures supplémentaires, déplacements professionnels et variable d'attrition.

La variable cible est :

| Variable | Description |
|---|---|
| `Attrition` | Indique si l'employé a quitté l'entreprise (`Yes`) ou non (`No`) |
| `Attrition_Flag` | Version numérique créée pour l'analyse et le modèle : `1 = départ`, `0 = reste` |

---

## 4. Outils et compétences mobilisés

| Outil / librairie | Utilisation dans le projet |
|---|---|
| Python | Analyse, nettoyage, statistiques, modélisation |
| pandas | Manipulation des données sous forme de DataFrame |
| matplotlib | Visualisations dans les notebooks |
| scipy | Tests statistiques |
| scikit-learn | Modélisation prédictive et pipeline de traitement |
| Streamlit | Application interactive finale |
| joblib | Sauvegarde des modèles |
| VS Code | Environnement de développement |
| GitHub | Versionnement et présentation du projet |

Documentation utile :

- pandas : https://pandas.pydata.org/docs/
- scikit-learn : https://scikit-learn.org/stable/
- Streamlit : https://docs.streamlit.io/

---

## 5. Structure du projet

```text
04_people_analytics_attrition/
|
|-- app/
|   |-- app.py
|
|-- data/
|   |-- raw/
|   |   |-- employee_attrition_raw.csv
|   |
|   |-- processed/
|       |-- employee_attrition_clean.csv
|       |-- statistical_results_categorical.csv
|       |-- statistical_results_numeric.csv
|       |-- feature_importance_random_forest.csv
|       |-- feature_importance_logistic_regression.csv
|
|-- models/
|   |-- attrition_random_forest_model.pkl
|   |-- attrition_logistic_model.pkl
|   |-- attrition_final_model.pkl
|   |-- model_features.pkl
|
|-- notebooks/
|   |-- 01_exploration_initiale.ipynb
|   |-- 02_analyse_exploratoire_rh.ipynb
|   |-- 03_analyse_statistique.ipynb
|   |-- 04_modelisation_prediction_attrition.ipynb
|
|-- visuals/
|   |-- streamlit_vue_ensemble.png
|   |-- streamlit_analyse_overtime.png
|   |-- streamlit_analyse_anciennete.png
|   |-- streamlit_simulateur_risque.png
|   |-- streamlit_recommandations.png
|   |-- modele_comparaison_scores.png
|   |-- importance_variables_random_forest.png
|   |-- facteurs_hausse_risque.png
|
|-- README.md
|-- requirements.txt
```

---

## 6. Méthodologie

### 6.1 Chargement et contrôle initial

Le premier notebook vérifie la structure du dataset :

- nombre de lignes ;
- nombre de colonnes ;
- types des variables ;
- valeurs manquantes ;
- doublons ;
- colonnes constantes.

Résultat après préparation :

| Indicateur | Valeur |
|---|---:|
| Nombre d'employés | 1 470 |
| Nombre de colonnes après nettoyage | 32 |
| Valeurs manquantes détectées | 0 |
| Doublons exacts détectés | 0 |

### 6.2 Nettoyage

Les principales opérations réalisées sont :

- création de la colonne `Attrition_Flag` ;
- suppression des colonnes constantes ou purement techniques ;
- sauvegarde d'un fichier propre dans `data/processed/`.

Colonnes supprimées :

| Colonne | Raison |
|---|---|
| `EmployeeCount` | Même valeur pour tous les employés |
| `Over18` | Même valeur pour tous les employés |
| `StandardHours` | Même valeur pour tous les employés |
| `EmployeeNumber` | Identifiant technique non utile pour l'analyse |

### 6.3 Analyse exploratoire

L'analyse exploratoire étudie notamment :

- le taux global d'attrition ;
- l'attrition par département ;
- l'impact des heures supplémentaires ;
- l'attrition par poste ;
- l'attrition selon la satisfaction ;
- l'attrition selon l'ancienneté ;
- l'attrition selon l'âge.

### 6.4 Analyse statistique

Deux types de tests ont été utilisés :

| Type de variable | Test utilisé | Exemple |
|---|---|---|
| Variables catégorielles | Test du khi-deux | `OverTime`, `JobRole`, `Department` |
| Variables numériques | Test de comparaison de moyennes | `Age`, `MonthlyIncome`, `YearsAtCompany` |

Ces tests ne prouvent pas une causalité. Ils indiquent seulement si une variable est statistiquement associée à l'attrition dans ce dataset.

### 6.5 Modélisation prédictive

Deux modèles ont été testés :

| Modèle | Rôle dans le projet |
|---|---|
| Régression logistique | Modèle simple, interprétable, intéressant pour expliquer les facteurs de risque |
| Random Forest | Modèle plus flexible, utile pour étudier l'importance des variables |

Le modèle final sauvegardé pour l'application Streamlit est la **régression logistique**, car son rappel est meilleur pour détecter les employés à risque.

---

## 7. Résultats principaux

### 7.1 KPI globaux

| KPI | Résultat |
|---|---:|
| Employés analysés | 1 470 |
| Employés partis | 237 |
| Employés restés | 1 233 |
| Taux d'attrition global | 16,12 % |

### 7.2 Principaux facteurs observés

Les analyses ont mis en évidence plusieurs signaux :

- les employés effectuant des heures supplémentaires présentent un taux d'attrition nettement plus élevé ;
- les salariés avec une ancienneté très faible sont davantage exposés au risque de départ ;
- certains postes comme `Sales Representative` ou `Laboratory Technician` ressortent comme plus sensibles ;
- l'âge, l'expérience, le salaire et la stabilité avec le manager sont associés au risque d'attrition ;
- la satisfaction et l'équilibre vie professionnelle / personnelle restent des dimensions importantes à surveiller.

---

## 8. Comparaison des modèles

| Modèle | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Régression logistique | 0.752 | 0.349 | 0.638 | 0.451 | 0.803 |
| Random Forest | 0.847 | 0.525 | 0.447 | 0.483 | 0.791 |

**Interprétation :**

Le Random Forest obtient une meilleure précision globale, mais la régression logistique détecte mieux les employés réellement partis grâce à un meilleur recall. Dans une logique RH préventive, il est préférable de limiter les faux négatifs, c'est-à-dire les employés à risque que le modèle ne détecterait pas.

---

## 9. Application Streamlit

L'application Streamlit transforme l'analyse en outil interactif.

Elle contient quatre pages :

| Page | Objectif |
|---|---|
| Vue d'ensemble | Afficher les KPI RH principaux |
| Analyse des facteurs | Explorer les taux d'attrition par variable |
| Simulateur de risque | Estimer le risque de départ d'un profil salarié |
| Recommandations RH | Traduire les résultats en actions concrètes |

### Lancer l'application

Depuis la racine du projet :

```bash
python -m streamlit run app/app.py
```

Puis ouvrir :

```text
http://localhost:8501
```

---

## 10. Captures à insérer

Les captures sont à placer dans le dossier `visuals/`.

### Vue d'ensemble Streamlit

![Vue d'ensemble Streamlit](visuals/streamlit_vue_ensemble.png)

### Analyse des heures supplémentaires

![Analyse OverTime](visuals/streamlit_analyse_overtime.png)

### Analyse de l'ancienneté

![Analyse ancienneté](visuals/streamlit_analyse_anciennete.png)

### Simulateur de risque

![Simulateur de risque](visuals/streamlit_simulateur_risque.png)

### Recommandations RH

![Recommandations RH](visuals/streamlit_recommandations.png)

### Comparaison des modèles

![Comparaison des modèles](visuals/modele_comparaison_scores.png)

### Importance des variables

![Importance des variables Random Forest](visuals/importance_variables_random_forest.png)

---

## 11. Recommandations métier

| Signal observé | Risque potentiel | Action recommandée |
|---|---|---|
| Heures supplémentaires fréquentes | Fatigue, surcharge, désengagement | Suivre la charge de travail et limiter les heures supplémentaires répétées |
| Forte attrition chez les nouveaux employés | Onboarding insuffisant ou mauvaise intégration | Renforcer le parcours d'intégration sur les 12 premiers mois |
| Faible satisfaction au travail | Perte de motivation | Mettre en place des enquêtes internes et des entretiens ciblés |
| Faible équilibre vie pro / vie perso | Risque de frustration ou burn-out | Favoriser la flexibilité et mieux répartir la charge |
| Déplacements fréquents | Fatigue et contraintes personnelles | Adapter la fréquence des déplacements quand c'est possible |
| Postes commerciaux et techniques plus exposés | Pression métier ou conditions spécifiques | Analyser les conditions propres à ces postes |

---

## 12. Limites du projet

Ce projet doit être interprété avec prudence :

- le dataset est un jeu de données d'entraînement, pas une donnée interne réelle ;
- les modèles détectent des associations, pas des causalités ;
- certaines variables peuvent être liées entre elles, par exemple salaire, poste et ancienneté ;
- un modèle RH ne doit jamais remplacer une analyse humaine ;
- dans un contexte réel, il faudrait compléter l'analyse avec des données qualitatives, des entretiens et une validation métier.

---

## 13. Conclusion

Ce projet montre une démarche complète de Data Analyst sur un cas RH : compréhension métier, nettoyage, exploration, statistiques, machine learning, interprétation et application interactive.

Il met en avant des compétences complémentaires aux autres projets du portfolio, notamment Python, pandas, scikit-learn, tests statistiques et Streamlit.

L'objectif final n'est pas de prédire parfaitement le départ d'un salarié, mais de construire un outil d'aide à l'analyse permettant d'identifier des signaux de risque et de proposer des actions RH concrètes.
