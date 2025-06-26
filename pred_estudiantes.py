import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pytorch_tabnet.tab_model import TabNetClassifier

# ----------------------------
# Cargar modelo TabNet entrenado
# ----------------------------
tabnet_model = TabNetClassifier()
tabnet_model.load_model("tabnet_model.zip")

# ----------------------------
# Cargar encoders
# ----------------------------
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# ----------------------------
# Cargar data original (no encodificada) para extraer valores únicos
# ----------------------------
df_raw = pd.read_csv("peru_student_enrollment_data_2023_clean.csv")

# Columnas categóricas (basado en tu info original)
cat_cols = [
    'tipo_matricula', 'genero', 'departamento', 'provincia', 'clasificacion',
    'campus', 'facultad', 'programa', 'turno', 'beneficios', 'modalidad', 'rango_edad'
]

# ----------------------------
# Título y descripción
# ----------------------------
st.set_page_config(
    page_title="Predicción de Riesgo Académico",
    layout="wide"
)
st.title("🎓 Predicción de Riesgo Académico Universitario")
st.markdown("Ingresa los datos del estudiante para predecir su **riesgo académico** con un modelo de aprendizaje automático basado en TabNet.")
# Crear layout en columnas
col_form, col_result = st.columns([2.5, 1], gap="large")

# Entrada en col1
with col_form:
    st.subheader("Datos del estudiante")
   

    user_input = {}

    # ----------------------------
    # Selectboxes anidados correctamente
    # ----------------------------
    departamento = st.selectbox("Departamento", sorted(df_raw["departamento"].dropna().unique()))
    df_dep = df_raw[df_raw["departamento"] == departamento]

    provincia = st.selectbox("Provincia", sorted(df_dep["provincia"].dropna().unique()))
    df_prov = df_dep[df_dep["provincia"] == provincia]

    campus = st.selectbox("Campus", sorted(df_prov["campus"].dropna().unique()))
    df_campus = df_prov[df_prov["campus"] == campus]

    facultad = st.selectbox("Facultad", sorted(df_campus["facultad"].dropna().unique()))
    df_facultad = df_campus[df_campus["facultad"] == facultad]

    programa = st.selectbox("Programa", sorted(df_facultad["programa"].dropna().unique()))

    # Guardar jerarquía en input
    user_input.update({
        "departamento": departamento,
        "provincia": provincia,
        "campus": campus,
        "facultad": facultad,
        "programa": programa,
    })

    # ----------------------------
    # Otros campos no anidados
    # ----------------------------
    for col in ['tipo_matricula', 'genero', 'clasificacion', 'turno', 'beneficios', 'modalidad', 'rango_edad']:
        opciones = sorted(df_raw[col].dropna().unique())
        user_input[col] = st.selectbox(col.replace("_", " ").capitalize(), opciones)

    user_input["num_cursos"] = st.number_input("Número de cursos matriculados", min_value=1, max_value=10, step=1)

# ----------------------------
# Botón de predicción
# ----------------------------
# Resultado en col2
with col_result:
    st.subheader("Resultado de la predicción")

    if st.button("Predecir"):
        input_df = pd.DataFrame([user_input])

        for col in encoders:
            if col in input_df.columns:
                input_df[col] = encoders[col].transform(input_df[col])

        columnas_modelo = [
            'tipo_matricula', 'genero', 'departamento', 'provincia', 'clasificacion',
            'campus', 'facultad', 'programa', 'turno', 'beneficios', 'modalidad',
            'rango_edad', 'num_cursos'
        ]
        input_df = input_df[columnas_modelo]

        probas = 1 - tabnet_model.predict_proba(input_df.values)[0][1]
        pred = 1 - int(probas > 0.5)

        # Mostrar resultado
        st.markdown(f"<h1 style='font-size: 42px; color: {'green' if pred == 1 else 'red'};'>"
                    f"{'Bajo' if pred == 1 else 'Alto'} riesgo</h1>", unsafe_allow_html=True)

        st.metric("Probabilidad de riesgo académico", f"{probas:.2%}")

# ----------------------------
# Estilos opcionales
# ----------------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stSelectbox label, .stNumberInput label {
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)