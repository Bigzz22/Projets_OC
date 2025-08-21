import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess_input
import time
import plotly.express as px
import pandas as pd
import os
import requests
from pathlib import Path


class_names = [
    'Maltese dog','Afghan hound','Scottish deerhound','Pomeranian',
    'Bernese mountain dog','Samoyed','Irish wolfhound','Shih Tzu',
    'Great Pyrenees','Leonberg'
]

MODEL_URLS = {
    "resnet": (
        "https://modelesreseff.blob.core.windows.net/models/resnet.h5?sp=r&st=2025-06-27T16:44:05Z&se=2025-07-28T00:44:05Z&spr=https&sv=2024-11-04&sr=b&sig=XS0shjAP4Flsa4mWWqFSO0h53wSLkaLMrULTYjmAQSo%3D"
    ),
    "effnet_weights": (
        "https://modelesreseff.blob.core.windows.net/models/effnet.weights.h5?sp=r&st=2025-06-27T16:33:04Z&se=2025-07-28T16:33:04Z&spr=https&sv=2024-11-04&sr=b&sig=iu2NSHjGcATr0r0KvKLTfvBWFzlZ6vPYSEmthNrdglI%3D"
    ),
}
LOCAL_DIR = "models"
CSV_PATH   = "data/image_labels.csv"

# 1) Chargement et cache des données
@st.cache_resource(show_spinner=False)
def load_models():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    # — ResNet50
    resnet_path = os.path.join(LOCAL_DIR, "resnet.h5")
    if not os.path.isfile(resnet_path):
        with st.spinner("Téléchargement de ResNet50…"):
            r = requests.get(MODEL_URLS["resnet"], stream=True)
            r.raise_for_status()
            with open(resnet_path, "wb") as f:
                for chunk in r.iter_content(16_384):
                    f.write(chunk)
    model_resnet = load_model(resnet_path)

    # — EfficientNetV2-S
    effnet_weights_path = os.path.join(LOCAL_DIR, "effnet.weights.h5")
    if not os.path.isfile(effnet_weights_path):
        with st.spinner("Téléchargement des poids EfficientNetV2-S…"):
            r = requests.get(MODEL_URLS["effnet_weights"], stream=True)
            r.raise_for_status()
            with open(effnet_weights_path, "wb") as f:
                for chunk in r.iter_content(16_384):
                    f.write(chunk)
    base = EfficientNetV2S(include_top=False, input_shape=(224,224,3), weights=None)
    x = GlobalAveragePooling2D()(base.output)
    out = Dense(len(class_names), activation="softmax")(x)
    model_effnet = Model(inputs=base.input, outputs=out)
    model_effnet.load_weights(effnet_weights_path)

    return model_resnet, model_effnet

@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)

resnet_model, effnet_model = load_models()
df = load_data()

st.title("Dashboard : Exploration de données et Prédiction")

# Palette accessible (viridis)
PALETTE = px.colors.sequential.Viridis

# Barre de Navigation
tab = st.sidebar.radio("Navigation", ["Exploration des données", "Prédiction de race"])

# 2) Exploration des données
if tab == "Exploration des données":
    st.markdown("---")
    st.header("Exploration des données")
    # Upload CSV
    if os.path.exists(CSV_PATH):
        # Affichage de la table
        st.markdown("---")
        st.subheader("Aperçu des labels")
        st.dataframe(df, use_container_width=True)

        # Graphique 1: Répartition des races
        st.markdown("---")
        st.subheader("Répartition des races")
        counts = df['true_label'].value_counts().reset_index()
        counts.columns = ['race', 'count']
        fig1 = px.bar(
            counts,
            x='race', y='count',
            color='race',
            color_discrete_sequence=PALETTE,
            title="Nombre d'images par race",
            labels={'race': 'Race', 'count': 'Nombre'}
        )
        fig1.update_layout(font=dict(size=16))
        st.plotly_chart(fig1, use_container_width=True)

        # Graphique 2: Distribution des races
        st.markdown("---")
        st.subheader("Distribution des races")
        fig_pie = px.pie(
            counts,
            names='race',
            values='count',
            title="Parts relatives des images par race",
            color_discrete_sequence=PALETTE
        )
        # Afficher uniquement le pourcentage, orientation horizontale
        fig_pie.update_traces(
            textinfo='percent',
            textposition='inside',
            insidetextorientation='horizontal'
        )
        fig_pie.update_layout(font=dict(size=16), showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

        # Affichage d'exemples
        st.markdown("---")
        st.subheader("Exemples d'images par race")
        for race in df['true_label'].unique():
            st.markdown(f"<p style='font-size:18px; font-weight:bold'>{race}</p>", unsafe_allow_html=True)


            folder   = Path("data/Images") / race
            examples = sorted([p.name for p in folder.iterdir()       
                            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                            ])[:3]

            cols = st.columns(len(examples) or 1)
            for col, fname in zip(cols, examples):
                img_path = folder / fname
                if img_path.exists():
                    img = Image.open(img_path).convert("RGB").resize((150, 150))
                    col.image(img, use_container_width=False)
                else:
                    col.text("Image non trouvée")

    else:
        st.error(f"Fichier '{CSV_PATH}' non trouvé. Veuillez placer 'image_labels.csv' dans le répertoire de travail.")


# 3) Comparaison des modèles et prédiction de race
else:
    st.markdown("---")
    st.header("Comparaison des modèles et prédiction")

    # Nombre de paramètres et couches
    st.markdown("---")
    st.subheader("Nombre de paramètres et couches")
    col_pr1, col_pr2 = st.columns(2)
    col_pr1.markdown(f"**ResNet50** : {resnet_model.count_params():,} paramètres, {len(resnet_model.layers)} couches")
    col_pr2.markdown(f"**EfficientNetV2S** : {effnet_model.count_params():,} paramètres, {len(effnet_model.layers)} couches")

    # F1-macro et Accuracy
    st.markdown("---")
    st.subheader("Précision (accuracy) et F1-macro")
    acc_r_val = 89.286
    f1_r_val  = 0.8708
    acc_e_val = 95.53
    f1_e_val  = 0.9321
    col_m1, col_m2 = st.columns(2)
    col_m1.markdown(f"**ResNet50** \n - Accuracy : {acc_r_val:.2f}% \n  - F1-macro : {f1_r_val:.3f}")
    col_m2.markdown(f"**EfficientNetV2S** \n - Accuracy : {acc_e_val:.2f}%  \n - F1-macro : {f1_e_val:.3f}")

    # Matrices
    st.markdown("---")
    st.subheader("Matrices de confusion")
    cm_cols = st.columns(2)
    cm_res_path = os.path.join("data", 'resnet_cm.png')
    cm_eff_path = os.path.join("data", 'effnet_cm.png')
    if os.path.exists(cm_res_path):
        cm_cols[0].image(cm_res_path, caption="ResNet50 Confusion Matrix", use_container_width=True)
    else:
        cm_cols[0].info("Matrice de confusion ResNet50 non trouvée")
    if os.path.exists(cm_eff_path):
        cm_cols[1].image(cm_eff_path, caption="EfficientNetV2S Confusion Matrix", use_container_width=True)
    else:
        cm_cols[1].info("Matrice de confusion EfficientNetV2S non trouvée")

    # Courbes
    st.markdown("---")
    st.subheader("Courbes d'entraînement de l'accuracy")
    cm_cols = st.columns(2)

    cm_res_path = os.path.join("data", 'resnet_courbe.png')
    cm_eff_path = os.path.join("data", 'effnet_courbe.png')
    if os.path.exists(cm_res_path):
        cm_cols[0].image(cm_res_path, caption="ResNet50 accuracy", use_container_width=True)
    else:
        cm_cols[0].info("Courbe de ResNet50 non trouvée")
    if os.path.exists(cm_eff_path):
        cm_cols[1].image(cm_eff_path, caption="EfficientNetV2S accuracy", use_container_width=True)
    else:
        cm_cols[1].info("Courbe d'EfficientNetV2S non trouvée")

    st.markdown("---")
    st.subheader("Prédiction")

    # Upload image
    uploaded = st.file_uploader("Téléversez une image de chien", type=['jpg','png','jpeg'])
    if uploaded:
        img = Image.open(uploaded).convert('RGB').resize((224,224))
        st.image(img, caption="Image téléchargée", use_container_width=False, width=300)

        img_array = np.array(img).astype('float32')
        x_resnet = resnet_preprocess_input(img_array.copy())
        x_resnet = np.expand_dims(x_resnet, axis=0)
        x_effnet = effnet_preprocess_input(img_array.copy())
        x_effnet = np.expand_dims(x_effnet, axis=0)

        x = np.array(img)[None, ...]

        fname = uploaded.name
        match = df[df['filename'].str.endswith(fname)]
        if not match.empty:
            true_label = match['true_label'].iloc[0]
            st.markdown(f"**True label :** {true_label}")
        else:
            true_label = None

        
        # Prédiction ResNet
        start_r = time.time()
        preds_r = resnet_model.predict(x, verbose=0)[0]
        time_r = time.time() - start_r
        idx_r = np.argmax(preds_r)
        label_r = class_names[idx_r]

        # Prédiction EfficientNet
        start_e = time.time()
        preds_e = effnet_model.predict(x, verbose=0)[0]
        time_e = time.time() - start_e
        idx_e = np.argmax(preds_e)
        label_e = class_names[idx_e]

        # Affichage comparatif
        col1, col2 = st.columns(2)
        col1.subheader("ResNet50")
        col1.markdown(f"**Prediction :** {label_r}\n\n**Temps inférence :** {time_r*1000:.1f} ms")
        col2.subheader("EfficientNetV2S")
        col2.markdown(f"**Prediction :** {label_e}\n\n**Temps inférence :** {time_e*1000:.1f} ms")

        # Graphique des temps d'inférence
        times_df = pd.DataFrame({
            'Model': ['ResNet50', 'EfficientNetV2S'],
            'Inference time (ms)': [time_r*1000, time_e*1000]
        })
        fig_time = px.bar(
            times_df,
            x='Model', y='Inference time (ms)',
            color='Model', color_discrete_sequence=PALETTE,
            title="Comparaison des temps d'inférence",
            labels={'Inference time (ms)': 'Temps (ms)'}
        )
        fig_time.update_layout(font=dict(size=16))
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Téléversez le fichier CSV pour commencer.")