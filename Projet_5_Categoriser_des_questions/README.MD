# 💡 Prédiction automatique de tags pour Stack Overflow

## 📌 Contexte du projet
**Stack Overflow** est l’une des plateformes les plus utilisées par les développeurs pour poser des questions techniques et trouver des solutions.  
Cependant, de nombreux utilisateurs débutants rencontrent des difficultés pour **attribuer les bons tags** à leurs questions, ce qui complique la recherche d’informations et la pertinence des résultats.

L’objectif de ce projet est de développer un **système de suggestion automatique de tags** à partir du contenu des questions, afin d’améliorer l’expérience utilisateur et la qualité des données sur la plateforme.

---

## 🎯 Objectifs du projet
- Extraire **50 000 questions** Stack Overflow via **StackExchange Data Explorer**.
- Nettoyer, explorer et prétraiter les données textuelles (titres, corps des questions, tags).
- Développer un **modèle de Machine Learning multi-label** pour suggérer les tags les plus pertinents.
- Tester l’utilisation de l’**API officielle Stack Exchange** pour récupérer et analyser des données récentes.
- Respecter les principes du **RGPD** en ne manipulant aucune donnée personnelle.

---

## 📂 Données utilisées
### **1. Extraction via StackExchange Data Explorer**
- Données extraites via requêtes SQL (50 000 questions).
- Variables principales :
  - **Title** → titre de la question.
  - **Body** → contenu complet de la question.
  - **Tags** → tags attribués par les utilisateurs.
  - **Score** → votes des internautes.
  - **ViewCount** → nombre de vues.
  - **FavoriteCount** → nombre d’ajouts en favoris.
  - **AnswerCount** → nombre de réponses.

### **2. API Stack Exchange**
- Test d’une requête sur 50 questions contenant le tag “python” avec un **score > 50**.
- Données extraites :
  - Date de création.
  - Titre de la question.
  - Tags attribués.
  - Score.

---

## 🧠 Approche méthodologique

### **1. Préparation des données**
- Suppression des doublons et lignes incomplètes.
- Nettoyage des balises HTML et du Markdown.
- Normalisation du texte :
  - Conversion en minuscules.
  - Suppression des caractères spéciaux et ponctuations.
  - Suppression des stopwords.
  - Lemmatisation.

---

### **2. Analyse exploratoire**
- Étude de la distribution des tags.
- Identification des tags les plus fréquents.
- Visualisations :
  - Nuages de mots des tags populaires.
  - Histogrammes des scores, vues et réponses.

---

### **3. Modélisation**
#### **Étapes principales :**
- Vectorisation des textes :
  - TF-IDF.
  - Embeddings via **Word2Vec** ou **FastText**.
- Modélisation multi-label :
  - **Logistic Regression One-vs-Rest**.
  - **Random Forest multi-output**.
  - **SVM**.
  - **Transformers** *(optionnel, pour amélioration)*.
- Évaluation des modèles :
  - F1-score macro et micro.
  - Hamming loss.
  - Top-k accuracy.

---

### **4. Intégration API**
- Utilisation de la bibliothèque **StackAPI** pour récupérer des questions récentes.
- Analyse des performances et de la documentation de l’API.

---

### **5. Respect du RGPD**
- Seules les données nécessaires au projet sont utilisées.
- Aucune information permettant d’identifier les auteurs n’est collectée.
- Les données sont traitées de manière anonymisée.

---

## 📊 Livrables
- **Notebook 1** → Extraction, nettoyage et analyse exploratoire.
- **Notebook 2** → Modélisation multi-label et évaluation.
- **Notebook 3** → Test de l’API Stack Exchange.
- **Présentation** → Synthèse des résultats et recommandations.

---

## ⚙️ Technologies utilisées
- **Langage :** Python 3.10
- **Bibliothèques principales :**
    - **Pandas / NumPy** → manipulation des données
    - **Matplotlib / Seaborn / Plotly** → visualisations
    - **Scikit-learn** → modélisation multi-label
    - **NLTK / SpaCy** → NLP et prétraitement
    - **TF-IDF / Word2Vec** → vectorisation
    - **StackAPI** → interrogation de l’API officielle
- **Outils :**
    - Jupyter Notebook
    - Git / GitHub

---

## 🚀 Résultats attendus
- Amélioration de la **précision de suggestion des tags**.
- Réduction des tags mal attribués pour les nouveaux utilisateurs.
- Une base technique solide pour intégrer l’algorithme dans l’interface Stack Overflow.

---

## 👤 Auteur
**Benjamin Bigot**  
📧 [benjamin.bigot22@gmail.com](mailto:benjamin.bigot22@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/benjamin-bigot-69b0581a3/)
---
