import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# 1. FUNÇÕES DE SUPORTE (UTILITÁRIOS)
# ==========================================

def guardar_modelo_e_metricas(objeto_modelo, metricas_dict, nome_protocolo, nome_algoritmo):
    """Guarda o modelo (.joblib) e o seu boletim de notas (.json) na respetiva pasta."""
    pasta = f'modelos_treinados/{nome_protocolo}/{nome_algoritmo}'
    os.makedirs(pasta, exist_ok=True)
    
    # Guardar Cérebro
    caminho_modelo = os.path.join(pasta, "modelo.joblib")
    joblib.dump(objeto_modelo, caminho_modelo)
    
    # Guardar Notas (Para a futura interface web)
    if metricas_dict:
        caminho_json = os.path.join(pasta, "metricas.json")
        with open(caminho_json, 'w') as f:
            json.dump(metricas_dict, f, indent=4)
            
    print(f"   ✅ {nome_algoritmo} guardado com sucesso!")

def calcular_metricas(y_real, y_previsto_binario):
    """Calcula o 'Boletim de Notas' padronizado para qualquer algoritmo."""
    cm = confusion_matrix(y_real, y_previsto_binario)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    
    return {
        "accuracy": round(accuracy_score(y_real, y_previsto_binario) * 100, 2),
        "precision": round(precision_score(y_real, y_previsto_binario, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_real, y_previsto_binario, zero_division=0) * 100, 2),
        "f1_score": round(f1_score(y_real, y_previsto_binario, zero_division=0) * 100, 2),
        "matriz_confusao": {"Verdadeiros_Negativos": int(tn), "Falsos_Positivos": int(fp), 
                            "Falsos_Negativos": int(fn), "Verdadeiros_Positivos": int(tp)}
    }

# ==========================================
# 2. INGESTÃO DE DADOS
# ==========================================

def carregar_dados_protocolo(caminho_pasta):
    """Lê CSVs, limpa dados e extrai os rótulos de forma dinâmica (À prova de bala)."""
    ficheiros_csv = glob.glob(os.path.join(caminho_pasta, "*.csv"))
    if not ficheiros_csv:
        raise FileNotFoundError(f"❌ Nenhum ficheiro .csv na pasta {caminho_pasta}")
    
    lista_dfs = []
    for ficheiro in ficheiros_csv:
        df = pd.read_csv(ficheiro)
        
        # 1. Tentar descobrir a coluna de Rótulo (Label) olhando para nomes comuns
        coluna_alvo = None
        for col in ['Label', 'label', 'Class', 'Attack_Label', 'Label_Category', 'Attack Type']:
            if col in df.columns:
                coluna_alvo = col
                break
        
        # 2. Criar o nosso Label_Binario (0 = Normal, 1 = Ataque)
        if coluna_alvo:
            # Converte para 0 se a palavra for 'benign', 'normal' ou 'normal traffic'
            df['Label_Binario'] = df[coluna_alvo].astype(str).apply(
                lambda x: 0 if x.strip().lower() in ['benign', 'normal', 'normal traffic'] else 1
            )
        else:
            # Fallback (A tática que usámos no MQTT, olhando para o nome do ficheiro)
            if 'normal' in ficheiro.lower():
                df['Label_Binario'] = 0
            else:
                df['Label_Binario'] = 1
                
        lista_dfs.append(df)
        
    dataset = pd.concat(lista_dfs, ignore_index=True)
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 3. Separar a solução (y) dos dados (X)
    y_binario = dataset['Label_Binario']
    
    # 4. O ESCUDO MÁGICO: Manter APENAS colunas matemáticas (inteiros e decimais)
    X = dataset.select_dtypes(include=[np.number])
    
    # Garantir que a resposta (Label_Binario) é removida do X para a IA não fazer batota!
    if 'Label_Binario' in X.columns:
        X = X.drop(columns=['Label_Binario'])
        
    return X, y_binario
# ==========================================
# 3. O MOTOR PRINCIPAL DE TREINO
# ==========================================

def treinar_fabrica(nome_protocolo, caminho_pasta):
    print(f"\n⚙️ A INICIAR FÁBRICA DE MODELOS PARA: {nome_protocolo} ⚙️")
    
    # Passo A: Preparação dos Dados
    X, y = carregar_dados_protocolo(caminho_pasta)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Selecionar Top 15 Features para otimização
    rf_temp = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1).fit(X_train, y_train)
    top_15 = pd.Series(rf_temp.feature_importances_, index=X.columns).nlargest(15).index.tolist()
    
    X_train = X_train[top_15]
    X_test = X_test[top_15]
    
    # Normalização Robusta
    scaler = RobustScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Guardar a Base (A Régua e as Features)
    os.makedirs(f'modelos_treinados/{nome_protocolo}/Base', exist_ok=True)
    joblib.dump(top_15, f'modelos_treinados/{nome_protocolo}/Base/features.joblib')
    joblib.dump(scaler, f'modelos_treinados/{nome_protocolo}/Base/scaler.joblib')

    # ---------------------------------------------------------
    # Passo B: Treinar Algoritmo 1 - Random Forest (Supervisionado)
    print("🧠 A treinar Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    
    rf_preds = rf.predict(X_test_scaled)
    rf_metricas = calcular_metricas(y_test, rf_preds)
    guardar_modelo_e_metricas(rf, rf_metricas, nome_protocolo, "RandomForest")

    # ---------------------------------------------------------
    # Passo C: Treinar Algoritmo 2 - Isolation Forest (Não Supervisionado)
    print("🧠 A treinar Isolation Forest...")
    # Aprende APENAS com tráfego normal
    X_train_normal = X_train_scaled[y_train == 0]
    
    iso_f = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    iso_f.fit(X_train_normal)
    
    iso_preds_raw = iso_f.predict(X_test_scaled)
    # Isolation Forest devolve -1 para anomalia e 1 para normal. Vamos converter para o nosso padrão (1=Ataque, 0=Normal)
    iso_preds = [1 if x == -1 else 0 for x in iso_preds_raw]
    
    iso_metricas = calcular_metricas(y_test, iso_preds)
    guardar_modelo_e_metricas(iso_f, iso_metricas, nome_protocolo, "IsolationForest")

    # ---------------------------------------------------------
    # Passo D: Treinar Algoritmo 3 - K-Means (Não Supervisionado)
    print("🧠 A treinar K-Means...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_train_scaled)
    
    kmeans_preds = kmeans.predict(X_test_scaled)
    # Como o K-Means não sabe o que é ataque, assumimos que o cluster com menos pacotes é a anomalia.
    cluster_anomalia = 1 if sum(kmeans_preds == 1) < sum(kmeans_preds == 0) else 0
    kmeans_preds_binario = [1 if x == cluster_anomalia else 0 for x in kmeans_preds]
    
    kmeans_metricas = calcular_metricas(y_test, kmeans_preds_binario)
    guardar_modelo_e_metricas(kmeans, kmeans_metricas, nome_protocolo, "KMeans")

    print(f"🏁 Fábrica de {nome_protocolo} terminada com sucesso!\n")


if __name__ == "__main__":
    # Testar o motor com o dataset que já temos estruturado:
    treinar_fabrica("CICIDS2017", "datasets/raw/CICIDS2017/") 
    # treinar_fabrica("MQTT", "datasets/raw/MQTT/")