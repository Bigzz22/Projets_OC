# dog_breed_prediction_app.py

import os
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input

# ----------------------------------------------------
# 1. Fonctions utilitaires
# ----------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_dog_model(model_path: str):
    """
    Charge et met en cache le modèle Keras depuis un fichier .h5.
    """
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Impossible de charger le modèle à partir de « {model_path} » : {e}")

def predict_breed(model, img: Image.Image, class_names: list[str]) -> tuple[str, float, np.ndarray]:
    """
    Prédit la race à partir d’une PIL.Image (RGB) :
      - model       : objet Keras chargé
      - img         : instance PIL.Image
      - class_names : liste de labels (ordre identique à la sortie du modèle)

    Retourne :
      - pred_label : label (str) prédit, ex. "Samoyed"
      - confidence : float (score de confiance = probabilité max)
      - probs      : np.ndarray shape=(num_classes,) (toutes les probabilités)
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    img_resized = img.resize((224, 224))
    x = keras_image.img_to_array(img_resized)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)  # shape = (1, 224, 224, 3)

    preds = model.predict(x, verbose=0)   # shape = (1, num_classes)
    probs = preds[0]                      # shape = (num_classes,)
    idx_max = np.argmax(probs)
    pred_label = class_names[idx_max].replace("_", " ")
    confidence = float(probs[idx_max])
    return pred_label, confidence, probs

# ----------------------------------------------------
# 2. Configuration de la page
# ----------------------------------------------------
st.set_page_config(
    page_title="🐶 Évaluation Streamlit – Prédiction race chiens",
    page_icon="🐕",
    layout="wide"
)

# ----------------------------------------------------
# 3. Titre et instructions
# ----------------------------------------------------
st.title("🐶 Évaluation de la race de chiens")

st.markdown("""
Cette application Streamlit :

1. Charge votre modèle (`mon_modele.h5`).
2. Lit le CSV `test_labels.csv` (colonnes obligatoires : `filename`, `true_label`).
3. Parcourt automatiquement **toutes** les images contenues dans le dossier `test_images/`.
4. Pour chaque image, affiche :
   - la vignette de la photo,
   - la race réelle (depuis le CSV),
   - la race prédite (par le modèle).
""")

# ----------------------------------------------------
# 4. Chargement du modèle
# ----------------------------------------------------
MODEL_PATH = "mon_modele.h5"
try:
    model = load_dog_model(MODEL_PATH)
except RuntimeError as e:
    st.error(f"❌ Erreur au chargement du modèle : {e}")
    st.stop()

# ----------------------------------------------------
# 5. Définition des classes / labels
# ----------------------------------------------------
breed_names = [
    "n02085936-Maltese_dog",
    "n02088094-Afghan_hound",
    "n02092002-Scottish_deerhound",
    "n02112018-Pomeranian",
    "n02107683-Bernese_mountain_dog",
    "n02111889-Samoyed",
    "n02090721-Irish_wolfhound",
    "n02086240-Shih-Tzu",
    "n02111500-Great_Pyrenees",
    "n02111129-Leonberg"
]
# (usage : .replace("_", " ") pour l’affichage si besoin)

# ----------------------------------------------------
# 6. Upload du CSV des labels réels
# ----------------------------------------------------
st.subheader("1. Uploadez votre fichier CSV de labels réels")
csv_file = st.file_uploader(
    label="Fichier CSV (colonnes obligatoires : `filename`, `true_label`)",
    type=["csv"],
    help="Par exemple : test_labels.csv"
)

if csv_file is not None:
    try:
        df_labels = pd.read_csv(csv_file, dtype=str)
    except Exception as e:
        st.error(f"❌ Impossible de lire le CSV : {e}")
        st.stop()

    # Vérification des colonnes
    if not {"filename", "true_label"}.issubset(df_labels.columns):
        st.error("Le CSV doit contenir EXACTEMENT les colonnes : `filename` et `true_label`.")
        st.stop()

    # Nettoyage léger
    df_labels["filename"]   = df_labels["filename"].astype(str).str.strip()
    df_labels["true_label"] = df_labels["true_label"].astype(str).str.strip()

    st.success(f"✅ CSV chargé ! Nombre d’entrées : {len(df_labels)}")

    # ----------------------------------------------------
    # 7. Vérification du dossier local test_images/
    # ----------------------------------------------------
    IMAGES_FOLDER = "test_images"
    if not os.path.isdir(IMAGES_FOLDER):
        st.error(f"⚠️ Le dossier `{IMAGES_FOLDER}` n’existe pas dans le répertoire courant.")
        st.stop()

    # Récupérer la liste des fichiers image
    all_files = sorted(os.listdir(IMAGES_FOLDER))
    image_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if len(image_files) == 0:
        st.warning(f"⚠️ Aucun fichier .jpg/.jpeg/.png trouvé dans `{IMAGES_FOLDER}`.")
        st.stop()

    st.info(f"🖼️ {len(image_files)} image(s) détectée(s) dans `{IMAGES_FOLDER}`.")

    # ----------------------------------------------------
    # 8. Bouton pour lancer l’évaluation en batch
    # ----------------------------------------------------
    if st.button("▶️ Lancer l’évaluation sur toutes les images"):
        with st.spinner("Traitement en cours…"):
            results = []  # liste de dicts pour le tableau récapitulatif

            # On parcourt chaque image du dossier
            for fname in image_files:
                # 8.1. Vérifier la présence dans le CSV
                match = df_labels[df_labels["filename"] == fname]
                if match.empty:
                    # Pas dans le CSV → on la ignore
                    continue

                true_label = match["true_label"].iloc[0]
                image_path = os.path.join(IMAGES_FOLDER, fname)

                # 8.2. Charger l’image avec PIL
                try:
                    img = Image.open(image_path)
                except Exception:
                    continue

                # 8.3. Prédiction
                pred_label, confidence, probs = predict_breed(model, img, breed_names)

                # 8.4. Stocker dans la liste pour le DataFrame
                results.append({
                    "filename"   : fname,
                    "true_label" : true_label,
                    "pred_label" : pred_label,
                    "confidence" : confidence,
                    "_pil_image" : img.copy()  # on stocke une copie PIL pour l’affichage individuel
                })

            # ----------------------------------------------------
            # 9. Affichage “image par image”
            # ----------------------------------------------------
            if len(results) == 0:
                st.warning("❌ Aucune image n’a pu être appariée au CSV ou n’est valide.")
            else:
                st.subheader("🔍 Résultats détaillés pour chaque image")

                # Pour chaque entrée, on affiche la vignette + labels
                # On crée un conteneur (expander) pour chaque image, afin de garder l’UI propre
                for entry in results:
                    fname   = entry["filename"]
                    true_lb = entry["true_label"]
                    pred_lb = entry["pred_label"]
                    img_pil = entry["_pil_image"]

                    # On affiche en expander (clic pour déplier) ou en colonne
                    with st.expander(f"🖼️ {fname}"):
                        # Afficher la vignette (largeur max = 300 px pour ne pas tout écraser)
                        st.image(img_pil, caption=f"{fname}", use_column_width=False, width=300)

                        # Affichage côte à côte de la race réelle et prédite
                        col1, col2 = st.columns(2)
                        col1.markdown(f"**🎯 Race réelle :** `{true_lb}`")
                        col2.markdown(f"**✅ Race prédite :** `{pred_lb}`")

                        # (Optionnel) On peut ajouter ici le barplot des probabilités si vous le souhaitez
                        # Par exemple :
                        # probs = entry["probs"]
                        # tri = np.argsort(probs)[::-1]
                        # labels = [b.replace("_"," ") for b in breed_names]
                        # sorted_probs = probs[tri]
                        # sorted_labels = [labels[i] for i in tri]
                        # plt.figure(figsize=(4,2.5))
                        # plt.barh(sorted_labels, sorted_probs)
                        # plt.gca().invert_yaxis()
                        # plt.tight_layout()
                        # st.pyplot()

else:
    st.info("⬆️ Commencez par uploader votre fichier CSV de labels réels (`test_labels.csv`).")