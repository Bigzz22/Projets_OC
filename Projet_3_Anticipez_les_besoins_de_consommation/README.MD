# 🌿 Projet Seattle — Prédiction des Émissions de CO₂ et Consommation Énergétique

## 📌 Contexte du projet
La ville de **Seattle** s’est fixé un objectif ambitieux : atteindre la **neutralité carbone d’ici 2050**.  
Pour y parvenir, la municipalité souhaite mieux comprendre et anticiper les **émissions de CO₂** et la **consommation énergétique** des **bâtiments non résidentiels**.

Des relevés précis ont été effectués en **2016**. Cependant, ces mesures sont **coûteuses** et ne peuvent pas être réalisées pour tous les bâtiments.  
L’objectif de ce projet est donc de **prédire les émissions de CO₂** et la **consommation totale d’énergie** pour les bâtiments dont ces données ne sont pas encore connues, en exploitant les **caractéristiques structurelles** des bâtiments.

---

## 🎯 Objectifs du projet
- **Analyser et nettoyer les données** pour comprendre les variables disponibles.
- **Réaliser une analyse exploratoire** pour identifier les corrélations clés.
- **Construire des modèles de Machine Learning** pour prédire :
  - Les **émissions de CO₂**.
  - La **consommation totale d’énergie**.
- **Évaluer la pertinence de la variable “ENERGY STAR Score”** :
  - Mesurer son impact sur les performances des modèles.
  - Déterminer si son calcul peut être évité à l’avenir.
- **Optimiser les modèles** pour maximiser la précision tout en évitant la fuite de données.

---

## 📂 Données utilisées
- **Source :** Données ouvertes de la ville de Seattle — [Seattle Open Data Portal](https://data.seattle.gov/)
- **Année :** 2016
- **Format :** CSV
- **Description des variables principales :**
    - **Caractéristiques structurelles** → surface, année de construction, type de bâtiment…
    - **Sources d’énergie** → proportions des différentes énergies utilisées.
    - **Consommation énergétique** → totale ou par type d’énergie.
    - **Émissions de CO₂** → mesurées pour certains bâtiments.
    - **ENERGY STAR Score** → indicateur global de performance énergétique.

---

## 🧠 Approche méthodologique

### **1. Exploration et préparation des données**
- Inspection des variables et compréhension des données.
- Nettoyage :
  - Gestion des **valeurs manquantes**.
  - Traitement des **valeurs aberrantes**.
- Transformation :
  - **Encodage** des variables catégorielles.
  - **Normalisation / standardisation** des variables numériques.
  - Passage au **log** pour certaines distributions fortement asymétriques.

---

### **2. Analyse exploratoire**
- Étude des corrélations entre variables structurelles, consommation et émissions.
- Visualisations clés :
  - Histogrammes pour les distributions.
  - Boxplots pour les valeurs aberrantes.
  - Heatmaps pour les corrélations.
  - Scatter plots pour identifier les tendances.

---

### **3. Modélisation**
- **Modèles testés :**
    - Régression linéaire.
    - Random Forest Regressor.
    - Gradient Boosting (XGBoost, LightGBM).
    - Réseaux de neurones simples.
- **Évaluation :**
    - Métriques : RMSE, MAE, R².
    - Validation croisée pour une estimation robuste des performances.
    - Comparaison des modèles avec et sans **ENERGY STAR Score**.

---

### **4. Optimisation des modèles**
- **Tuning des hyperparamètres** via `GridSearchCV` et `RandomizedSearchCV`.
- Sélection du **meilleur modèle** sur la base des performances globales.

---

### **5. Résultats attendus**
- Jeu de données **propre et prêt pour la modélisation**.
- Modèle performant capable de **prédire les émissions de CO₂** et la **consommation énergétique**.
- Évaluation claire de l’apport réel du **ENERGY STAR Score**.
- Recommandations pour la ville de Seattle sur l’**utilité d’automatiser le calcul de certaines variables**.

---

## 📊 Visualisations produites
- Heatmap des corrélations.
- Histogrammes des principales variables.
- Boxplots pour les variables continues.
- Courbes de comparaison des modèles.

---

## ⚙️ Technologies utilisées
- **Langage :** Python
- **Bibliothèques principales :**
    - **Pandas / NumPy** → manipulation des données
    - **Matplotlib / Plotly** → visualisations interactives
    - **Scikit-learn** → modélisation et tuning
- **Outils :**
    - Kaggle Notebook
    - Git / GitHub

---

## 📌 Contraintes et précautions
- **Pas de fuite de données** : ne pas utiliser les consommations réelles pour les bâtiments à prédire.
- **Automatisation** : prévoir un pipeline réutilisable sur de nouvelles données.
- **Lisibilité** : les visualisations doivent être adaptées à un public non technique.

---

## 👤 Auteur
**Benjamin Bigot**  
📧 [benjamin.bigot22@gmail.com](mailto:benjamin.bigot22@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/benjamin-bigot-69b0581a3/)
