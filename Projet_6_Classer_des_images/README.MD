# 🐶 Classification d’images de races de chiens

## 📌 Contexte du projet
L’association **Le Refuge**, qui recueille et protège des animaux, souhaite accélérer l’indexation de sa base de données d’images de chiens.  
Actuellement, les bénévoles n’ont pas le temps de classer manuellement les photos selon la race des animaux, ce qui ralentit leur organisation.  

L’objectif de ce projet est de développer un **algorithme de classification supervisée** capable de **prédire la race d’un chien à partir d’une photo**, afin d’automatiser et fiabiliser ce travail.

---

## 🎯 Objectifs
- Mettre en place un pipeline complet de **prétraitement et augmentation d’images**.  
- Construire et comparer deux approches :
  1. **Un CNN développé from scratch**, en s’inspirant de modèles existants.
  2. **Le transfer learning** avec des réseaux pré-entraînés (e.g. VGG16, ResNet, Inception).  
- Évaluer les performances en termes de précision, temps de calcul et robustesse.  
- Fournir une **démonstration** du moteur de classification (via notebook ou application Streamlit).  

---

## 📂 Données
- **Jeu de données utilisé** : [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)  
- **Taille** : 20 580 images réparties sur 120 classes de races de chiens.  
- **Format** : Images couleur, tailles variables.  

*(Pour des raisons de ressources, des tests peuvent être réalisés sur un sous-ensemble de 3 races uniquement.)*

---

## 🧠 Méthodologie

### **1. Prétraitement des images**
- Redimensionnement et normalisation.
- Techniques de data augmentation :
  - Rotation
  - Mirroring (flip horizontal/vertical)
  - Cropping
  - Égalisation / whitening

### **2. Approche 1 : CNN développé from scratch**
- Conception d’un réseau convolutionnel personnalisé.  
- Entraînement sur un sous-ensemble de données.  
- Optimisation des hyperparamètres :
  - Nombre de filtres et taille des kernels.  
  - Fonction d’activation (ReLU, softmax).  
  - Optimiseur (Adam, SGD).  

### **3. Approche 2 : Transfer Learning**
- Utilisation de modèles pré-entraînés sur ImageNet :  
  - **VGG16**  
  - **ResNet50**  
  - **InceptionV3**  
- Réentraînement des dernières couches pour prédire uniquement les races de chiens.  
- Ajustement des poids via fine-tuning avec un **faible learning rate**.  

### **4. Évaluation**
- Métriques principales :
  - Accuracy top-1 et top-5.
  - Matrice de confusion par classes.
- Comparaison :
  - CNN from scratch vs Transfer Learning.
  - Temps d’entraînement vs performance.

---

## 🚀 Démonstration
- **Option 1** → Notebook Jupyter interactif pour tester le modèle sur des images.  
- **Option 2** → Application **Streamlit** permettant de charger une photo et de prédire la race du chien en direct.  

---

## 📊 Résultats attendus
- Précision significativement améliorée grâce au **transfer learning**.  
- Modèle déployable pour aider l’association **Le Refuge** à indexer automatiquement ses images.  
- Pipeline reproductible, utilisable à plus grande échelle sur l’ensemble du dataset.  

---

## ⚙️ Technologies utilisées
- **Langage** : Python 3.10  
- **Bibliothèques principales** :
  - **TensorFlow / Keras** → CNN, transfer learning  
  - **NumPy / Pandas** → gestion des données  
  - **Matplotlib / Seaborn** → visualisations  
  - **OpenCV / PIL** → traitement des images  
- **Outils** :
  - Jupyter Notebook  
  - Google Colab (accès GPU gratuit)  
  - Streamlit (démonstration)  

---

## 👤 Auteur
**Benjamin Bigot**  
📧 [benjamin.bigot22@gmail.com](mailto:benjamin.bigot22@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/benjamin-bigot-69b0581a3/)
---
