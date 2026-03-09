🐾 Classification Canine : Preuve de Concept (PoC) SOTA
Ce projet présente une amélioration significative d'un système de classification d'images de chiens en remplaçant une architecture classique par un modèle issu de la recherche récente. Réalisé pour le compte de DataSpace, ce projet démontre une méthodologie complète allant de la veille technologique au déploiement cloud. 

🎯 Objectifs
Veille Technologique : Identifier une architecture plus performante que les baselines standards (ResNet). 

Performance : Améliorer l'accuracy et réduire le temps d'inférence sur GPU. 
+1

Industrialisation : Développer un dashboard interactif et le déployer via Docker. 
+1

📊 Jeu de Données : Stanford Dogs
Le projet s'appuie sur le dataset Stanford Dogs, comprenant plus de 20 000 images réparties par races. 

Prétraitement : Uniformisation de la taille (224×224×3) et normalisation des pixels. 
+1

Data Augmentation : Utilisation d'Albumentations (Equalization, ShiftScaleRotate, Flip, RandomBrightnessContrast) pour renforcer la robustesse du modèle. 

🧠 Architecture & Innovation
La solution repose sur le modèle EfficientNetV2S, basé sur l'article pivot "EfficientNetV2: Smaller Models and Faster Training" (Tan & Le, 2021). 
+1

Pourquoi ce choix ?

Architecture optimisée : Utilisation de blocs Fused-MBConv (remplaçant les MBConv classiques) pour accélérer l'entraînement et l'inférence. 
+1

Adaptabilité : Plus performant sur des datasets de taille moyenne qu'un Vision Transformer (ViT). 

Efficacité : Moins de paramètres (20,3M contre 23,6M pour ResNet-50) et moins de FLOPs. 
+1

📈 Résultats Comparatifs (Baseline vs SOTA)
La preuve de concept valide la supériorité de l'approche choisie :

Métrique	Baseline (ResNet-50)	SOTA (EfficientNetV2S)
Accuracy Top-1	89.29%	95.54%
F1-macro Score	0.8708	0.9321
Temps d’inférence (GPU)	219,8 ms	168,3 ms
Poids (Paramètres)	23,6 M	20,3 M
Note : Bien que plus performant sur GPU, le modèle présente un temps d'inférence légèrement plus long sur CPU par rapport à la baseline. 

🚀 Déploiement & Dashboard
Une interface interactive a été développée pour permettre aux utilisateurs de tester le modèle en conditions réelles. 

Framework : Streamlit 

Conteneurisation : Image Docker hébergée sur DockerHub. 

Cloud : Déploiement sur Azure App Service. 

Fonctionnalités : Exploration de la distribution des races et prédiction en temps réel sur image uploadée.
