import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os

# Configuração da Página
st.set_page_config(page_title="IoT Anomaly Detection Dashboard", layout="wide")

st.title("🛡️ Dashboard de Análise de Deteção de Anomalias IoT")

# ==========================================
# 1. Função Dinâmica para Carregar Métricas
# ==========================================

@st.cache_data
def load_all_metrics(base_dir="modelos_treinados"):
    """
    Percorre a estrutura: modelos_treinados/<protocolo>/<modelo>/metricas.json
    """
    all_data = {}
    
    if not os.path.exists(base_dir):
        st.error(f"A pasta '{base_dir}' não foi encontrada.")
        return {}

    # 1º Nível: Listar subpastas de Protocolos/Datasets (ex: MQTT, CICIDS2017)
    protocolos = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for protocolo in protocolos:
        all_data[protocolo] = {}
        protocolo_path = os.path.join(base_dir, protocolo)
        
        # 2º Nível: Listar subpastas de Modelos (ex: random_forest, kmeans)
        modelos = [m for m in os.listdir(protocolo_path) if os.path.isdir(os.path.join(protocolo_path, m))]
        
        for modelo in modelos:
            # 3º Nível: O ficheiro JSON chama-se sempre 'metricas.json'
            json_path = os.path.join(protocolo_path, modelo, "metricas.json")
            
            if os.path.exists(json_path):
                # Formatar o nome do modelo para a interface gráfica (ex: 'random_forest' -> 'Random Forest')
                nome_modelo_formatado = modelo.replace('_', ' ').title()
                
                # Ler o ficheiro json
                with open(json_path, 'r', encoding='utf-8') as f:
                    all_data[protocolo][nome_modelo_formatado] = json.load(f)
                    
    return all_data

# ==========================================
# 2. Interface Principal da Dashboard
# ==========================================

# Carregar dados
metricas_globais = load_all_metrics()

if not metricas_globais:
    st.warning("Nenhum dado encontrado. Verifica se a pasta 'modelos_treinados' está na mesma diretoria do 'app.py'.")
else:
    # Sidebar: Seleção do Protocolo
    st.sidebar.header("Configurações de Visualização")
    protocolo_selecionado = st.sidebar.selectbox("Selecione o Protocolo:", list(metricas_globais.keys()))

    # Filtrar dados para o protocolo selecionado
    dados_protocolo = metricas_globais[protocolo_selecionado]

    # --- Separadores de Visualização ---
    tab1, tab2 = st.tabs(["📊 Visão por Modelo", "🔄 Comparação de Modelos"])

    with tab1:
        st.subheader(f"Métricas Detalhadas: {protocolo_selecionado}")
        
        # Só mostrar se existirem modelos para este protocolo
        if dados_protocolo:
            modelo_escolhido = st.selectbox("Escolha o Modelo:", list(dados_protocolo.keys()))
            
            m = dados_protocolo[modelo_escolhido]
            
            col1, col2, col3, col4 = st.columns(4)
            # Usamos o .get() para não dar erro caso o JSON não tenha alguma destas métricas
            col1.metric("Accuracy", f"{m.get('accuracy', 0):.2f}%")
            col2.metric("Precision", f"{m.get('precision', 0):.2f}%")
            col3.metric("Recall", f"{m.get('recall', 0):.2f}%")
            col4.metric("F1-Score", f"{m.get('f1_score', 0):.2f}%")
        else:
            st.info("Nenhum modelo treinado encontrado para este protocolo.")

    with tab2:
        st.subheader("Comparação de Performance entre Modelos")
        
        if dados_protocolo:
            lista_comp = []
            for mod, stats in dados_protocolo.items():
                for metrica in ['accuracy', 'precision', 'recall', 'f1_score']:
                    # Só adicionamos se a métrica existir no JSON
                    if metrica in stats:
                        lista_comp.append({
                            "Modelo": mod,
                            "Métrica": metrica.replace('_', ' ').title(),
                            "Valor": stats[metrica]
                        })
            
            if lista_comp:
                df_comp = pd.DataFrame(lista_comp)
                
                # Gráfico
                fig = px.bar(df_comp, x="Métrica", y="Valor", color="Modelo", 
                             barmode="group",
                             title=f"Comparação para Protocolo {protocolo_selecionado}")
                
                fig.update_traces(texttemplate='%{y:.2f}%', textposition='auto')

                # Ajustar eixo Y para percentagem (0 a 100)
                fig.update_layout(yaxis=dict(range=[0, 100], ticksuffix='%'))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Métricas insuficientes para gerar a comparação.")