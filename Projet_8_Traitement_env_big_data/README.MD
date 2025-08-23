# ☁️ Traitement Big Data sur le Cloud — Projet Fruits!

## 📌 Contexte du projet
La start-up **Fruits!** est une jeune entreprise innovante en **AgriTech**.  
Son objectif est de développer des **solutions intelligentes pour la récolte des fruits** tout en préservant la biodiversité.

Dans un premier temps, l’entreprise souhaite se faire connaître via une **application mobile** qui permettrait à un utilisateur de **prendre une photo d’un fruit** et d’obtenir automatiquement des informations sur celui-ci.  
Cette application servira également à préparer la **première version du moteur de classification d’images**.

Pour supporter les futures évolutions et la croissance du volume de données, la start-up souhaite mettre en place une **architecture Big Data scalable sur le cloud AWS**.

---

## 🎯 Objectifs du projet
- Mettre en place une **chaîne de traitement Big Data** robuste et évolutive.
- Créer et configurer une **infrastructure AWS EMR** pour traiter un grand volume d’images.
- Développer un script **PySpark** pour préparer les données et optimiser les performances.
- Intégrer :
    1. Le **broadcast des poids du modèle TensorFlow** sur l’ensemble des nœuds du cluster.
    2. Une **réduction de dimension PCA** en PySpark pour accélérer les traitements.
- Garantir la **conformité RGPD** en configurant les serveurs dans l’Union Européenne.
- Minimiser les **coûts liés à AWS** en optimisant l’utilisation de l’instance EMR.

---

## 📂 Données utilisées
- **Source :** [Fruit Images Dataset](https://www.kaggle.com/moltean/fruits)
- **Taille :** ~44 000 images
- **Nombre de classes :** 131 types de fruits
---

## 🧠 Approche méthodologique

### **1. Architecture Big Data**
- Mise en place d’une **infrastructure AWS EMR** :
    - **Amazon S3** : stockage des images et résultats.
    - **Amazon EMR** : traitement distribué via PySpark.
    - **AWS IAM** : gestion des permissions.
    - **Databricks** *(optionnel)* : alternative pour les traitements avancés.

---

### **2. Traitements en PySpark**
#### **Prétraitement des images :**
- Chargement des données depuis **Amazon S3**.
- Redimensionnement et normalisation des images.
- Encodage des labels.

#### **Broadcast des poids TensorFlow :**
- Diffusion des **poids du modèle pré-entraîné** sur les nœuds du cluster pour optimiser l’inférence.
- Amélioration significative des performances lors des prédictions distribuées.

#### **Réduction de dimension (PCA) :**
- Utilisation de la **librairie MLlib** de PySpark.
- Réduction de la taille des vecteurs d’images pour :
    - Diminuer le temps de calcul.
    - Optimiser la mémoire sur les nœuds.

---

### **3. Optimisation des coûts AWS**
- Développement et tests locaux → **serveur local** pour limiter les coûts.
- Utilisation ponctuelle de l’instance **AWS EMR** uniquement pour les phases critiques.
- Limitation des coûts totaux à **moins de 10 €**.

---

## 📊 Livrables
- **Script PySpark principal** → prétraitement des images et PCA.
- **Script TensorFlow** → broadcast des poids sur le cluster.
- **Documentation technique** → déploiement de l’infrastructure EMR.
- **Démonstration** → traitement Big Data opérationnel sur AWS.

---

## ⚙️ Technologies utilisées
- **Langage :** Python 3.10
- **Framework Big Data :**
    - **PySpark** → traitement distribué.
    - **MLlib** → PCA et machine learning distribué.
- **Deep Learning :**
    - **TensorFlow / Keras** → gestion des poids du modèle.
- **Cloud :**
    - **AWS EMR** → cluster Big Data.
    - **AWS S3** → stockage des données.
    - **AWS IAM** → gestion des droits d’accès.
    - **Databricks** *(optionnel)* → alternative pour le développement collaboratif.
- **Outils :**
    - Jupyter Notebook
    - Git / GitHub

---

## 👤 Auteur
**Benjamin Bigot**  
📧 [benjamin.bigot22@gmail.com](mailto:benjamin.bigot22@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/benjamin-bigot-69b0581a3/)

---
