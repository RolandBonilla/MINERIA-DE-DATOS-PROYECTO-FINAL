import streamlit as st
import pandas as pd
import joblib

# Configuración de página
st.set_page_config(
    page_title="Predicción de Churn Telecom",
    page_icon="📱",
    layout="centered"
)

st.title("📱 Sistema de Evaluación de Riesgo de Churn")
st.markdown("### Clasificación de Fuga de Clientes en Telecomunicaciones")
st.write("Ingrese los parámetros del cliente para estimar la probabilidad de abandono:")

# Cargar el modelo entrenado
try:
    pipeline = joblib.load('modelo_churn.pkl')
except Exception as e:
    st.error(f"⚠️ No se pudo cargar el archivo 'modelo_churn.pkl': {e}")
    st.info("Por favor ejecuta primero 'python analysis.py' para generar el modelo.")
    st.stop()

# Formulario interactivo
with st.form("form_cliente"):
    st.subheader("📋 Datos Contractuales y de Servicio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        contract = st.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
        tenure = st.number_input("Antigüedad (meses)", min_value=0, max_value=72, value=6)
        internet = st.selectbox("Servicio de Internet", ["Fiber optic", "DSL", "No"])
        monthly_charges = st.number_input("Cargo Mensual ($)", min_value=18.0, max_value=120.0, value=75.0)
        total_charges = st.number_input("Cargo Total ($)", min_value=0.0, max_value=9000.0, value=450.0)

    with col2:
        tech_support = st.selectbox("Soporte Técnico", ["No", "Yes", "No internet service"])
        online_sec = st.selectbox("Seguridad en Línea", ["No", "Yes", "No internet service"])
        payment = st.selectbox("Método de Pago", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = st.selectbox("Facturación Electrónica", ["Yes", "No"])
        senior = st.selectbox("Adulto Mayor", [0, 1])

    btn_evaluar = st.form_submit_button("🔍 Evaluar Riesgo de Cliente")

if btn_evaluar:
    data_dict = {
        'gender': 'Female', 'SeniorCitizen': senior, 'Partner': 'No', 'Dependents': 'No',
        'tenure': tenure, 'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': internet, 'OnlineSecurity': online_sec, 'OnlineBackup': 'No',
        'DeviceProtection': 'No', 'TechSupport': tech_support, 'StreamingTV': 'No', 'StreamingMovies': 'No',
        'Contract': contract, 'PaperlessBilling': paperless, 'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }
    
    df_in = pd.DataFrame([data_dict])
    
    # Feature Engineering dinámico
    df_in['Cargo_Promedio_Mensual'] = df_in.apply(
        lambda r: r['TotalCharges'] / r['tenure'] if r['tenure'] > 0 else r['MonthlyCharges'], axis=1
    )
    servicios = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    n_serv = sum([1 for s in servicios if df_in[s].iloc[0] == 'Yes'])
    df_in['Indice_Fidelidad'] = (n_serv * (df_in['tenure'].iloc[0] + 1)) / (df_in['MonthlyCharges'].iloc[0] + 1)
    
    # Predicción
    prob = pipeline.predict_proba(df_in)[0][1]
    
    st.divider()
    st.subheader("📊 Resultado del Diagnóstico:")
    
    if prob >= 0.5:
        st.error(f"⚠️ **ALTO RIESGO DE ABANDONO (CHURN)**")
        st.metric("Probabilidad de Fuga", f"{prob*100:.1f}%")
        st.warning("💡 **Recomendación Comercial:** Iniciar campaña de retención de inmediato. Migrar al cliente a contrato anual ofreciendo un 15% de descuento en cargo mensual.")
    else:
        st.success(f"✅ **CLIENTE RETENIDO / FIDELIZADO**")
        st.metric("Probabilidad de Fuga", f"{prob*100:.1f}%")
        st.info("💡 **Recomendación Comercial:** Cliente estable. Promocionar servicios adicionales de streaming.")