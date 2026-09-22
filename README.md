# 📊 Predicción de Fuga de Clientes en Telecomunicaciones Mediante Técnicas de Minería de Datos

> **Informe del Componente Práctico-Experimental (CPE)**  
> **Universidad Estatal Amazónica (UEA)**  
> **Carrera:** Tecnologías de la Información  
> **Asignatura:** Minería de Datos  
> **Estudiante:** Roland Joel Bonilla Paredes  
> **Docente:** Ing. Delfín Bernabé Ortega Tenezaca Mgs.  
> **Fecha:** Septiembre 2026  

---

## 📌 Resumen del Proyecto

La fuga de clientes (*customer churn*) representa uno de los mayores desafíos financieros para el sector de las telecomunicaciones. Este proyecto aplica un flujo metodológico de minería de datos basado en el estándar **CRISP-DM** sobre un conjunto de **7,043 registros reales** de clientes (*Telco Customer Churn dataset*). 

Se compararon tres algoritmos de aprendizaje automático supervisado (**Regresión Logística**, **Árbol de Decisión** y **Random Forest**), evaluados mediante validación cruzada estratificada ($K=5$) y métricas estandarizadas (Accuracy, Precision, Recall, F1-Score y AUC-ROC). El modelo óptimo (Regresión Logística, AUC-ROC = 0.841) fue extraído, serializado y desplegado mediante una **API REST (FastAPI)** y una **Web App interactiva (Streamlit)** para apoyar la toma de decisiones comerciales preventivas.

---

## 🚀 Estructura del Repositorio

El repositorio está organizado conforme a los requerimientos oficiales de la práctica:

```text
Proyecto-Final-Mineria-de-Datos/
│
├── data/
│   └── telecom_churn_dataset.csv     # Dataset base (7043 registros, 21 variables)
│
├── figs/                             # Visualizaciones generadas automáticamente
│   ├── fig1_hist_tenure.png          # Histograma de antigüedad del cliente
│   ├── fig2_box_monthlycharges.png   # Boxplot de cargos mensuales
│   ├── fig3_bar_churn_contract.png   # Tasa de fuga según tipo de contrato
│   ├── fig4_roc_curves.png           # Curvas ROC comparativas
│   ├── fig5_confusion_matrix.png     # Matriz de confusión del mejor modelo
│   └── fig6_feature_importance.png   # Top 10 variables con mayor incidencia
│
├── analysis.py                       # Pipeline de minería de datos (EDA, limpieza, modelado)
├── generate_data.py                  # Script de construcción y preparación del dataset
├── notebook_churn.ipynb              # Cuaderno interactivo Jupyter Notebook 
├── modelo_churn.pkl                  # Modelo exportado binario serializado 
├── api.py                            # Código fuente de la API REST en FastAPI 
├── app.py                            # Código fuente de la Web App en Streamlit 
├── report_data.json                  # Registro numérico de métricas y estadísticas
├── requirements.txt                  # Dependencias de Python necesarias
└── README.md                         # Documentación general del proyecto

📈 Metodología y Resultados (DIKW)
El proyecto se estructura bajo la jerarquía DIKW (Datos, Información, Conocimiento, Sabiduría):

1. Datos (Data)
Volumen: 7,043 registros y 21 variables.
Calidad de datos: 11 valores nulos en TotalCharges (0.16%) e imputados por mediana; 22 registros duplicados (0.31%) eliminados, fijando una base depurada de 7,021 registros.

2. Información (Information)
Variables Derivadas (Feature Engineering):
Cargo_Promedio_Mensual: TotalCharges / tenure (suaviza la facturación histórica).
Indice_Fidelidad: (N° Servicios x (tenure + 1)) / (MonthlyCharges + 1).
Hallazgo exploratorio clave: Los clientes con contrato mes a mes (Month-to-month) presentan una tasa de deserción superior al 40%, frente a menos del 5% en contratos anuales.

3. Conocimiento (Knowledge)
Comparativa de Modelos (Conjunto de Prueba - 1,405 registros):
Modelo	Accuracy	Precision	Recall	F1-Score	AUC-ROC	CV AUC-ROC (K=5)
Regresión Logística	0.8007	0.6575	0.5161	0.5783	0.8406	0.8466 ± 0.015
Random Forest	0.7972	0.6582	0.4866	0.5595	0.8398	0.8477 ± 0.017
Árbol de Decisión	0.7907	0.6523	0.4489	0.5318	0.8302	0.8219 ± 0.017
Factores de Incidencia Dominantes: Tipo de contrato (Month-to-month), antigüedad del cliente (tenure) y el Cargo_Promedio_Mensual.

4. Sabiduría (Wisdom) - Despliegue Tecnológico
Modelo Serializado: Exportado como modelo_churn.pkl usando joblib.
API REST (FastAPI): Expone el endpoint /predict para integraciones entre sistemas.
Web App (Streamlit): Panel interactivo para el equipo comercial con recomendaciones prescriptivas de retención.

🛠️ Instalación y Ejecución

1. Clonar el repositorio e instalar dependencias
git clone https://github.com/RolandBonilla/MINERIA-DE-DATOS-PROYECTO-FINAL-.git
cd Proyecto-Final-Mineria-de-Datos-
pip install -r requirements.txt

2. Ejecutar el Pipeline de Minería de Datos
Para ejecutar la exploración, preprocesamiento, entrenamiento y exportar el modelo modelo_churn.pkl:
python analysis.py

3. Ejecutar la API REST (FastAPI)
python -m uvicorn api:app --reload


4. Ejecutar la Web App Interactiva (Streamlit)
python -m streamlit run app.py


📚 Referencias Bibliográficas

Akbar, T. A. R., & Apriono, C. (2023). Machine learning predictive models analysis on telecommunications service churn rate. Green Intelligent Systems and Applications, 3(1), 22–34. https://doi.org/10.53623/gisa.v3i1.249
Joseph, V. R. (2022). Optimal ratio for data splitting. Statistical Analysis and Data Mining, 15(4), 531–538. https://doi.org/10.1002/sam.11583
Wagh, S. K., et al. (2023). Customer churn prediction in telecom sector using machine learning techniques. Results in Control and Optimization, 14, 100342. https://doi.org/10.1016/j.rico.2023.100342
Zhou, Y., Chen, W., & Sun, X. (2023). Early warning of telecom enterprise customer churn based on ensemble learning. PLOS ONE, 18(10), e0292466. https://doi.org/10.1016/j.pone.0292466

© 2026 Roland Joel Bonilla Paredes - Universidad Estatal Amazónica (UEA)
