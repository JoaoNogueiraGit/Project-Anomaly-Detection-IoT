import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="IoT Anomaly Detection Dashboard", layout="wide")

st.title("🛡️ Dashboard de Deteção de Anomalias IoT")

# 1. Carregar Modelo e Dados
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

st.sidebar.header("Configurações")
modelo_selecionado = st.sidebar.selectbox("Escolha o Modelo", ["Random Forest", "Isolation Forest"])

# Espaço para Upload de Dados
uploaded_file = st.sidebar.file_uploader("Carregar tráfego (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Pré-visualização dos Dados", df.head())

    # Exemplo de Gráfico de Resultados (Dashboards)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuição de Classes")
        fig = px.pie(df, names='label', title="Ataques vs Tráfego Normal")
        st.plotly_chart(fig)

    with col2:
        st.subheader("Análise de Features")
        # Aqui podes usar a lógica do teu dataset_analyzer.py
        fig2 = px.histogram(df, x="packet_length", color="label", barmode="overlay")
        st.plotly_chart(fig2)

# 3. Métricas de Treino (Lendo os teus ficheiros JSON de resultados)
st.divider()
st.header("📈 Performance do Modelo")
# Simulação de métricas que já tens no projeto
metrics = {"Accuracy": 0.77, "Precision": 0.75, "Recall": 0.80}
st.json(metrics)