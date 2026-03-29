# Features seleccionadas automaticamente pelo IoT IDS Feature Analyzer
# Score baseado em Cohen's d + diferença % + significância estatística (Mann-Whitney U)
# NOTA: Limiares fixos não produziram resultados — usado percentil 70 do score (0.266) como limiar adaptativo.

FEATURES_EXCELENTES = ['bwd_num_psh_flags', 'fwd_num_psh_flags', 'bwd_num_rst_flags', 'bwd_num_pkts', 'fwd_num_pkts', 'bwd_std_pkt_len', 'bwd_min_pkt_len', 'bwd_max_pkt_len']
FEATURES_BOAS       = []
FEATURES_MODERADAS  = ['fwd_num_rst_flags', 'bwd_min_iat', 'fwd_min_iat', 'bwd_std_iat', 'fwd_std_iat', 'bwd_max_iat', 'fwd_max_iat', 'fwd_min_pkt_len']
FEATURES_ML         = FEATURES_EXCELENTES + FEATURES_BOAS + FEATURES_MODERADAS  # recomendado para treino

# Se FEATURES_ML estiver vazio (dataset com baixa separação), usar:
# FEATURES_ML = FEATURES_EXCELENTES + FEATURES_BOAS + FEATURES_MODERADAS

# Uso:
# X = df[FEATURES_ML]
# y = df['is_attack']
# clf = RandomForestClassifier().fit(X, y)
# clf = IsolationForest().fit(X)
