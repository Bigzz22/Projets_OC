🛍️ Segmentation des clients Olist

📌 Contexte du projet

Olist est une entreprise brésilienne qui propose une solution de vente sur les marketplaces en ligne.
L’équipe Marketing souhaite mieux comprendre les comportements des clients afin d’optimiser les campagnes de communication et d’améliorer l’expérience utilisateur.

Notre mission consiste à segmenter les clients en groupes homogènes selon leurs comportements d’achat, leur satisfaction et leurs informations personnelles, à partir d’une base de données anonymisée fournie par Olist.

⸻

🎯 Objectifs
	•	Réaliser une analyse exploratoire des données pour identifier les tendances et comportements des clients.
	•	Mettre en place une segmentation client pertinente et exploitable pour l’équipe Marketing.
	•	Comparer différentes méthodes de clustering afin de trouver la plus adaptée.
	•	Fournir une recommandation de fréquence de mise à jour du modèle de segmentation afin de maintenir sa pertinence dans le temps.
	•	Livrer un code propre, documenté et conforme aux standards PEP8.

⸻

📂 Données

Le jeu de données fourni par Olist contient des informations issues de plusieurs tables :
	•	Historique des commandes : dates, montants, statuts, délais de livraison.
	•	Produits achetés : catégories, prix, quantités.
	•	Commentaires clients : satisfaction, notes.
	•	Informations clients : localisation, état, ville.

Période couverte : janvier 2017 → aujourd’hui.

⸻

🧠 Méthodologie

1. Analyse exploratoire (EDA)
	•	Étude des distributions des variables.
	•	Détection et traitement des valeurs manquantes.
	•	Identification des variables pertinentes pour la segmentation.

2. Segmentation client
	•	Méthodes testées :
	•	Segmentation RFM (Récence, Fréquence, Montant).
	•	Clustering non supervisé :
	•	K-Means
	•	DBSCAN
	•	Agglomerative Clustering
	•	Évaluation des méthodes :
	•	Score de silhouette
	•	Davies-Bouldin Index
	•	Interprétabilité pour le Marketing

3. Stabilité et mise à jour des segments
	•	Étude de la dérive des segments dans le temps.
	•	Simulation de différentes fréquences de recalcul.
	•	Recommandation finale sur la périodicité de mise à jour.

⸻

🛠️ Technologies utilisées
	•	Langage : Python 
	•	Librairies principales :
	•	Pandas, NumPy → manipulation des données
	•	Matplotlib, Plotly → visualisations
	•	Scikit-learn → clustering, métriques, simulations

⸻

🚀 Résultats attendus
	•	Une segmentation claire et exploitable pour l’équipe Marketing.
	•	Une recommandation sur la fréquence de mise à jour.
	•	Une base technique solide pour automatiser le recalcul des segments à l’avenir.
