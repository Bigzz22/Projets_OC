# 🍎 Projet Open Food Facts — Santé Publique France

## 📌 Contexte du projet
L'agence **Santé Publique France** souhaite améliorer la qualité de la base de données **Open Food Facts**, qui fournit des informations nutritionnelles détaillées sur des produits alimentaires.  
Actuellement, l'ajout de produits dans la base nécessite de remplir de nombreux champs, ce qui entraîne souvent des **erreurs de saisie** et **des valeurs manquantes**.  

Notre mission est de :
1. **Nettoyer et explorer le jeu de données** Open Food Facts.
2. **Évaluer la faisabilité** de suggérer automatiquement des valeurs manquantes.
3. Préparer la base pour une future application d’**auto-complétion intelligente**.

---

## 🎯 Objectifs
- Identifier les variables pertinentes pour les analyses.
- Nettoyer les données : gestion des valeurs manquantes et des valeurs aberrantes.
- Automatiser le nettoyage pour gérer les évolutions futures du dataset.
- Produire des **analyses univariées** et **multivariées** pour mieux comprendre les relations entre variables.
- Tester la faisabilité de **suggestions automatiques** de valeurs manquantes.
- Respecter les principes du **RGPD** dans la manipulation des données.

---

## 📂 Données utilisées
- **Source :** [Open Food Facts](https://world.openfoodfacts.org/data)
- **Format :** CSV
- **Description :**  
Le jeu de données contient des milliers de produits et se compose de 4 grandes sections :
1. **Informations générales** → nom, date de modification, etc.
2. **Tags** → catégorie du produit, localisation, origine, etc.
3. **Ingrédients** → composition et additifs éventuels.
4. **Informations nutritionnelles** → quantité en grammes d’un nutriment pour 100g de produit.

---

## 🧠 Approche et méthodologie

### **1. Traitement et nettoyage des données**
- Sélection des variables pertinentes pour l'analyse.
- Détection des **valeurs manquantes** avec plusieurs méthodes :
  - Suppression des lignes/colonnes peu informatives.
  - Imputation par **moyenne/médiane** pour les variables numériques.
  - Imputation par **mode** ou **référence externe** pour les variables catégorielles.
  - Remplissage par **0** pour certaines catégories spécifiques (*ex. boissons, sauces, glaces…*).
- Détection et traitement des **valeurs aberrantes**.

### **2. Automatisation du pipeline**
- Création de fonctions réutilisables pour :
  - Identifier les valeurs manquantes.
  - Appliquer les méthodes d’imputation.
  - Gérer les valeurs aberrantes.
- Conception d’un code robuste capable de gérer un dataset mis à jour.

### **3. Analyse exploratoire**
- **Analyse univariée** → histogrammes, boxplots, diagrammes circulaires.
- **Analyse multivariée** → corrélations, tests statistiques, ACP si nécessaire.
- Production de visualisations lisibles pour un public néophyte.

### **4. Étude de faisabilité**
- Sélection d’une variable avec **plus de 50% de valeurs manquantes**.
- Évaluation de la pertinence des modèles d’imputation.
- Démonstration de la possibilité d’**auto-complétion intelligente**.

---

## 📊 Visualisations produites
- Histogrammes pour la distribution des variables.
- Boxplots pour l’étude des valeurs aberrantes.
- Nuages de points pour les corrélations.
- Heatmaps pour visualiser les relations multivariées.

---

## ⚙️ Technologies utilisées
- **Langage :** Python
- **Bibliothèques principales :**
    - **Pandas / NumPy** → manipulation des données
    - **Matplotlib / Plotly** → visualisations interactives
    - **Scikit-learn** → prétraitement et imputation
- **Outils :**
    - Kaggle Notebook
    - Git / GitHub

---

## 📌 Résultats attendus
- Jeu de données **nettoyé** et **prêt** pour l’analyse.
- Méthodes d’imputation **testées et documentées**.
- **Rapport d’exploration** clair et accessible.
- **Présentation finale** adaptée à un public néophyte.

---

## 👤 Auteur
**Benjamin Bigot**  
📧 [benjamin.bigot22@gmail.com](mailto:benjamin.bigot22@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/benjamin-bigot-69b0581a3/)
