# Predicción de Riesgo Académico Universitario

Este proyecto aplica modelos de aprendizaje automático para **predecir el riesgo académico** de estudiantes universitarios a partir de sus características de matrícula, datos académicos y socioeconómicos.

## Objetivo

Desarrollar una herramienta que permita, a partir de datos tabulares, identificar estudiantes con **alto riesgo académico**, permitiendo a las instituciones tomar decisiones anticipadas de intervención.

---

## Modelos usados

- **Multilayer Perceptron (MLP)** optimizado con Keras Tuner
- **TabNet (PyTorch)** con búsqueda de hiperparámetros vía Optuna

Se seleccionó como modelo final **TabNet** por su capacidad de interpretar datos tabulares, manejar desbalance de clases y obtener mejor rendimiento (F1 score más alto).

---

## Estructura del proyecto

- `app.py`: Aplicación en Streamlit para realizar predicciones.
- `tabnet_model.zip`: Modelo TabNet entrenado y guardado.
- `encoders.pkl`: Encoders usados para transformar variables categóricas.
- `peru_student_enrollment_data_2023_clean.csv`: Dataset limpio sin codificar (usado para opciones desplegables).
- `README.md`: Documentación general del proyecto.

---

## Variables utilizadas

| Variable         | Descripción                                           | Tipo         |
|------------------|--------------------------------------------------------|--------------|
| tipo_matricula   | Tipo de matrícula (regular, extraordinaria, etc.)     | Categórica   |
| genero           | Género del estudiante                                  | Categórica   |
| departamento     | Departamento de residencia                             | Categórica   |
| provincia        | Provincia de residencia                                | Categórica   |
| clasificacion    | Clasificación socioeconómica                           | Categórica   |
| campus           | Campus donde estudia                                   | Categórica   |
| facultad         | Facultad académica                                     | Categórica   |
| programa         | Programa o carrera universitaria                       | Categórica   |
| turno            | Turno de estudios (mañana, noche, etc.)               | Categórica   |
| beneficios       | Cantidad de beneficios otorgados                       | Numérica     |
| modalidad        | Modalidad de estudios (presencial, virtual, etc.)      | Categórica   |
| rango_edad       | Rango de edad del estudiante                           | Categórica   |
| num_cursos       | Número de cursos matriculados                          | Numérica     |
| riesgo_academico | Variable objetivo: 1 si hay riesgo, 0 si no            | Binaria      |

---

## Cómo ejecutar

1. Clona el repositorio:

```bash
git clone https://github.com/tuusuario/riesgo-academico.git
cd riesgo-academico