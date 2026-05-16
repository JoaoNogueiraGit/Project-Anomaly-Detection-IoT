import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
from thefuzz import process 
import pipeline 
import shutil

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
                # Formatar o nome do modelo para a interface gráfica
                nome_modelo_formatado = modelo.replace('_', ' ').title()
                
                # Ler o ficheiro json
                with open(json_path, 'r', encoding='utf-8') as f:
                    all_data[protocolo][nome_modelo_formatado] = json.load(f)
                    
    return all_data

# ==========================================
# 2. Interface Principal da Dashboard
# ==========================================

metricas_globais = load_all_metrics()

# Sidebar: Seleção do Protocolo (Movida para cima para servir todas as tabs)
st.sidebar.header("Configurações Globais")

if metricas_globais:
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚠️ Zona de Perigo")
    
    # Lista os datasets disponíveis para eliminação
    dataset_para_eliminar = st.sidebar.selectbox(
        "Selecionar dataset para apagar:",
        options=list(metricas_globais.keys()),
        key="delete_select"
    )
    
    # Checkbox de segurança para o utilizador confirmar que sabe o que está a fazer
    confirmar_remocao = st.sidebar.checkbox(
        f"Confirmo que quero apagar os dados de '{dataset_para_eliminar}'", 
        key="confirm_check"
    )
    
    if st.sidebar.button("🗑️ Apagar Dataset e Modelos", type="primary", disabled=not confirmar_remocao):
        pasta_alvo = os.path.join("modelos_treinados", dataset_para_eliminar)
        
        if os.path.exists(pasta_alvo):
            try:
                # 1. Remover a pasta física e ficheiros (.joblib, .json)
                shutil.rmtree(pasta_alvo)
                
                # 2. Limpar a cache do Streamlit para forçar a releitura do disco
                load_all_metrics.clear()
                
                st.sidebar.success(f"Dados de '{dataset_para_eliminar}' removidos com sucesso!")
                
                # 3. Forçar o Streamlit a recarregar a interface imediatamente
                st.rerun()
                
            except Exception as e:
                st.sidebar.error(f"Erro ao apagar ficheiros: {e}")
else:
    st.sidebar.markdown("---")
    st.sidebar.info("Nenhum dataset disponível para eliminação.")

# --- NOVO: 3 Separadores em vez de 2 ---
tab1, tab2, tab3 = st.tabs(["📊 Visão por Modelo", "🔄 Comparação de Modelos", "🚀 Treinar Novo Modelo"])

with tab1:
    st.subheader("Métricas Detalhadas")
    if metricas_globais:
        protocolo_selecionado = st.selectbox("Selecione o Protocolo:", list(metricas_globais.keys()), key="tab1_prot")
        dados_protocolo = metricas_globais[protocolo_selecionado]
        
        if dados_protocolo:
            modelo_escolhido = st.selectbox("Escolha o Modelo:", list(dados_protocolo.keys()))
            
            m = dados_protocolo[modelo_escolhido]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{m.get('accuracy', 0):.2f}%")
            col2.metric("Precision", f"{m.get('precision', 0):.2f}%")
            col3.metric("Recall", f"{m.get('recall', 0):.2f}%")
            col4.metric("F1-Score", f"{m.get('f1_score', 0):.2f}%")
        else:
            st.info("Nenhum modelo treinado encontrado para este protocolo.")
    else:
        st.warning("Nenhum dado encontrado. Treine um modelo no separador 'Treinar Novo Dataset'.")

with tab2:
    st.subheader("Comparação de Performance entre Modelos")
    if metricas_globais:
        protocolo_selecionado_t2 = st.selectbox("Selecione o Protocolo:", list(metricas_globais.keys()), key="tab2_prot")
        dados_protocolo = metricas_globais[protocolo_selecionado_t2]
        
        if dados_protocolo:
            # 1. Preparar o DataFrame original com todos os dados
            lista_comp = []
            for mod, stats in dados_protocolo.items():
                for metrica in ['accuracy', 'precision', 'recall', 'f1_score']:
                    if metrica in stats:
                        lista_comp.append({
                            "Modelo": mod,
                            "Métrica": metrica.replace('_', ' ').title(),
                            "Valor": stats[metrica]
                        })
            
            if lista_comp:
                df_comp = pd.DataFrame(lista_comp)
                
                # --- NOVO: Controlos de Filtragem (Multiselect) ---
                st.markdown("##### 🎛️ Filtrar Dados")
                modelos_disponiveis = df_comp['Modelo'].unique().tolist()
                metricas_disponiveis = df_comp['Métrica'].unique().tolist()
                
                col_filtro1, col_filtro2 = st.columns(2)
                
                modelos_selecionados = col_filtro1.multiselect(
                    "Modelos a mostrar:", 
                    options=modelos_disponiveis, 
                    default=modelos_disponiveis # Por defeito, mostra todos
                )
                
                metricas_selecionadas = col_filtro2.multiselect(
                    "Métricas a mostrar:", 
                    options=metricas_disponiveis, 
                    default=metricas_disponiveis # Por defeito, mostra todas
                )
                
                # 2. Criar um novo DataFrame apenas com o que o utilizador selecionou
                df_filtrado = df_comp[
                    (df_comp['Modelo'].isin(modelos_selecionados)) & 
                    (df_comp['Métrica'].isin(metricas_selecionadas))
                ]
                
                # Só desenha o gráfico e mostra exportação se houver dados após o filtro
                if not df_filtrado.empty:
                    # Gráfico (agora usa o df_filtrado)
                    fig = px.bar(df_filtrado, x="Métrica", y="Valor", color="Modelo", 
                                 barmode="group",
                                 title=f"Comparação para Protocolo {protocolo_selecionado_t2}")
                    
                    fig.update_traces(texttemplate='%{y:.2f}%', textposition='auto')
                    fig.update_layout(yaxis=dict(range=[0, 100], ticksuffix='%'))
                    st.plotly_chart(fig, use_container_width=True)

                    # Exportar Relatório (também usa o df_filtrado)
                    st.markdown("---")
                    st.subheader("Exportar Relatório Personalizado")
                    
                    formato_escolhido = st.radio(
                        "Selecione o formato de exportação:",
                        options=["CSV", "JSON"],
                        horizontal=True,
                        key="export_radio"
                    )
                    
                    if formato_escolhido == "CSV":
                        dados_exportar = df_filtrado.to_csv(index=False).encode('utf-8')
                        extensao = "csv"
                        mime_type = "text/csv"
                    else:
                        dados_exportar = df_filtrado.to_json(orient='records', indent=4).encode('utf-8')
                        extensao = "json"
                        mime_type = "application/json"
                    
                    st.download_button(
                        label=f"📥 Descarregar Resultados em {formato_escolhido}",
                        data=dados_exportar,
                        file_name=f'comparacao_{protocolo_selecionado_t2}_filtrada.{extensao}',
                        mime=mime_type
                    )
                else:
                    st.info("⚠️ Selecione pelo menos um Modelo e uma Métrica para visualizar e exportar os dados.")
            else:
                st.warning("Métricas insuficientes para gerar a comparação.")

# O Motor de Treino
with tab3:
    st.subheader("Importar Dataset Próprio e Treinar")

    if st.session_state.get('treino_concluido', False):
        nome_treinado = st.session_state.get('nome_recente', 'o seu dataset')
        st.success(f"✅ Modelos para '{nome_treinado}' treinados com sucesso!")
        st.balloons()
        # "Rasgar a nota" para que a animação não se repita a cada clique no site
        st.session_state.treino_concluido = False
    
    st.write("Faça upload do seu ficheiro CSV. O sistema irá automaticamente limpar os dados, extrair as melhores variáveis matemáticas e treinar 3 modelos (Random Forest, Isolation Forest e K-Means).")
    
    uploaded_file = st.file_uploader("Importar o seu dataset (.csv)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, comment = "#")
        colunas_disponiveis = df.columns.tolist()
        
        with st.expander("Pré-visualizar Dataset"):
            st.dataframe(df.head())

        # Tentar adivinhar a coluna de Label (Fuzzy Matching)
        melhor_palpite, score = process.extractOne("Label", colunas_disponiveis)
        index_padrao = colunas_disponiveis.index(melhor_palpite) if score > 80 else 0

        with st.form("form_treino"):
            nome_novo_modelo = st.text_input("Dê um nome a este conjunto de dados (ex: IoT_Rede_Casa):", value="Custom_Dataset")
            
            coluna_alvo = st.selectbox(
                "Qual destas colunas indica se o tráfego é Normal (0) ou Ataque (1)? (Label)",
                options=colunas_disponiveis,
                index=index_padrao
            )
            
            botao_treinar = st.form_submit_button("🚀 Iniciar Treino de Modelos")

        if botao_treinar:
            if nome_novo_modelo.strip() == "":
                st.error("Por favor, insira um nome válido para o conjunto de modelos.")
            else:
                with st.spinner("A limpar dados e a treinar algoritmos... Isto pode demorar alguns minutos dependendo do tamanho do dataset."):
                    try:
                        # Chama a função adaptada no pipeline.py
                        pipeline.treinar_fabrica_via_web(nome_novo_modelo, df, coluna_alvo)
                        
                        # Limpa a cache do Streamlit para ler os novos JSONs criados
                        load_all_metrics.clear()
                        
                        st.session_state.treino_concluido = True
                        st.session_state.nome_recente = nome_novo_modelo
                        
                        # O recarregamento acontece imediatamente a seguir
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Ocorreu um erro durante o treino: {e}")