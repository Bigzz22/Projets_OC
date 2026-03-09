# 🐶 Classification d’images de races de chiens – Proof of Concept
## 📌 Contexte du projet

Ce projet a été réalisé dans le cadre d’un test technique pour DataSpace, une entreprise spécialisée dans la conception de solutions de data science et d’intelligence artificielle.
L’objectif est de réaliser une veille technologique en machine learning, puis de mettre en place une preuve de concept (Proof of Concept) afin d’améliorer les performances d’un modèle existant de classification d’images.
Le projet consiste à développer un modèle capable de prédire la race d’un chien à partir d’une image, puis à comparer :
un modèle baseline (ResNet-50)
un modèle plus récent issu de la recherche : EfficientNetV2
Les résultats sont ensuite présentés via un dashboard interactif déployé sur le cloud.

---

## 🎯 Objectifs
Réaliser une veille technologique sur les modèles récents en deep learning.
Implémenter un nouvel algorithme publié récemment.
Comparer ses performances avec un modèle de référence plus ancien.
Évaluer les modèles selon :
la précision
le score F1
le temps d’inférence
la complexité du modèle.
Développer un dashboard interactif permettant de tester le modèle de classification.

---

## 📂 Données
Jeu de données utilisé : Stanford Dogs Dataset
Source : http://vision.stanford.edu/aditya86/ImageNetDogs/
Caractéristiques du dataset
20 580 images
120 races de chiens
Images de tailles variables
Organisation : 1 dossier par race
Les images présentent une grande variabilité :
angles de prise de vue
conditions d’éclairage
arrière-plans
positions des chiens
Cette diversité rend la tâche de classification plus réaliste mais aussi plus complexe.

---

## 🧠 Méthodologie
1️⃣ Prétraitement des images
Avant l’entraînement des modèles, plusieurs étapes de préparation des données ont été appliquées :
Transformations
Resize : uniformiser la taille des images
Normalization : centrer et réduire les pixels
Equalization : améliorer le contraste local
Data Augmentation
Afin d’améliorer la capacité de généralisation du modèle :
ShiftScaleRotate → translation, zoom, rotation
Horizontal / Vertical Flip → invariance spatiale
RandomBrightnessContrast → variations de luminosité
MedianBlur → robustesse au bruit
### 🤖 Approche 1 : Modèle baseline (ResNet-50)
Le modèle de référence utilisé est ResNet-50, un réseau convolutionnel profond très utilisé en vision par ordinateur.
Le modèle est initialisé avec des poids pré-entraînés sur ImageNet.
Hyperparamètres
Optimizer : Adam
Learning rate : 1e-5
Batch size : 32
Epochs : 30
Dropout : 0.5
### 🚀 Approche 2 : Modèle récent – EfficientNetV2
Suite à la veille technologique, le modèle retenu est EfficientNetV2.
Article de référence
EfficientNetV2: Smaller Models and Faster Training
Mingxing Tan & Quoc V. Le – Google Research (2021)
https://arxiv.org/pdf/2104.00298v3
Pourquoi ce modèle ?
Meilleure efficacité computationnelle
Temps d’entraînement réduit
Architecture optimisée pour les datasets de taille moyenne
Meilleure précision globale
Contrairement aux Vision Transformers, EfficientNetV2 est souvent plus performant lorsque la taille du dataset est limitée.

---

## ⚙️ Nouveaux concepts introduits
Fused-MBConv
EfficientNetV2 remplace les blocs MBConv traditionnels par des Fused-MBConv :
meilleure efficacité GPU
entraînement plus rapide
Training-Aware Neural Architecture Search (NAS)
L’architecture a été optimisée via une méthode de Neural Architecture Search prenant en compte :
- la précision
- la vitesse d’entraînement

---

## 📊 Résultats
Comparaison entre les deux modèles :
Modèle	ResNet-50	EfficientNetV2S
Accuracy Top-1	89.29 %	95.54 %
Accuracy Top-3	97.32 %	99.55 %
F1-macro score	0.8708	0.9321
Nombre de paramètres	23.6M	20.3M
Temps d’inférence GPU	219.8 ms	168.3 ms
Conclusion
Le modèle EfficientNetV2S :
- améliore significativement la précision
- utilise moins de paramètres
- possède un temps d’inférence plus rapide sur GPU
Cependant, sur CPU, le temps d’inférence peut être légèrement supérieur.

---

## 📊 Dashboard
Un dashboard interactif a été développé avec Streamlit pour démontrer la preuve de concept.
Fonctionnalités principales :
exploration du dataset
upload d’une image
prédiction de la race du chien
affichage des top-3 prédictions

---

## ☁️ Déploiement
Le dashboard a été déployé sur le cloud via :
Docker
DockerHub
Azure App Service
Pipeline de déploiement
Création de l’image Docker
Push de l’image sur DockerHub
Déploiement sur Azure App Service

---

## ⚙️ Technologies utilisées
Langage
Python 3.10
Bibliothèques principales
TensorFlow / Keras → deep learning
NumPy / Pandas → manipulation des données
Matplotlib / Seaborn → visualisations
OpenCV / PIL → traitement d’images
Outils
Jupyter Notebook
Streamlit
Docker
Azure App Service
DockerHub

---

## 👤 Auteur
Benjamin Bigot
📧 benjamin.bigot22@gmail.com
🔗 https://www.linkedin.com/in/benjamin-bigot-69b0581a3/
