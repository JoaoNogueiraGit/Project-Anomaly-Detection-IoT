"""
IoT IDS Feature Analyzer
========================
Analisa datasets de tráfego IoT (ex: MQTT-IoT-IDS2020) com múltiplos
ficheiros CSV, compara tráfego normal vs ataque, e identifica as features
mais discriminativas para treino de modelos ML (Random Forest, Isolation Forest, etc.)

Uso:
    # Pasta com CSVs (label automática pelo nome do ficheiro)
    python iot_feature_analyzer.py --path ./dados/

    # Ficheiro único com coluna 'is_attack'
    python iot_feature_analyzer.py --path ./dados/biflow_mqtt_bruteforce.csv

    # Com opções
    python iot_feature_analyzer.py --path ./dados/ --label is_attack --threshold 50 --export ./resultados/
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
import importlib, subprocess

# ─── Dependências ──────────────────────────────────────────────────────────────
def ensure(pkg, import_as=None):
    name = import_as or pkg
    try:
        importlib.import_module(name)
    except ImportError:
        print(f"[setup] A instalar {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg, imp in [("colorama", None), ("tabulate", None), ("scipy", None)]:
    ensure(pkg, imp)

import pandas as pd
import numpy as np
from pathlib import Path
from colorama import Fore, Style, init as colorama_init
from tabulate import tabulate
from scipy import stats as scipy_stats

colorama_init(autoreset=True)


# ─── Colunas a ignorar (identificadores, não features de rede) ─────────────────
NON_FEATURES = {
    "ip_src", "ip_dst", "src_ip", "dst_ip",
    "prt_src", "prt_dst", "src_port", "dst_port",
    "proto", "protocol", "timestamp", "time",
    "flow_id", "label", "is_attack", "attack",
    "class", "category", "type",
}

# Grupos de features (para análise organizada)
FEATURE_GROUPS = {
    "Volumes de Pacotes": ["fwd_num_pkts", "bwd_num_pkts"],
    "Inter-Arrival Time (IAT)": [
        "fwd_mean_iat", "bwd_mean_iat",
        "fwd_std_iat",  "bwd_std_iat",
        "fwd_min_iat",  "bwd_min_iat",
        "fwd_max_iat",  "bwd_max_iat",
    ],
    "Comprimento de Pacotes": [
        "fwd_mean_pkt_len", "bwd_mean_pkt_len",
        "fwd_std_pkt_len",  "bwd_std_pkt_len",
        "fwd_min_pkt_len",  "bwd_min_pkt_len",
        "fwd_max_pkt_len",  "bwd_max_pkt_len",
    ],
    "Bytes": ["fwd_num_bytes", "bwd_num_bytes"],
    "Flags TCP": [
        "fwd_num_psh_flags", "bwd_num_psh_flags",
        "fwd_num_rst_flags", "bwd_num_rst_flags",
        "fwd_num_urg_flags", "bwd_num_urg_flags",
    ],
}

BANNER = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════╗
║          IoT IDS Feature Analyzer — MQTT-IoT-IDS2020           ║
║   Normal vs Ataque · Discriminação · Ranking para ML           ║
╚══════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""


# ─── Helpers ───────────────────────────────────────────────────────────────────
def ok(msg):      print(f"  {Fore.GREEN}✔{Style.RESET_ALL}  {msg}")
def warn(msg):    print(f"  {Fore.YELLOW}⚠{Style.RESET_ALL}  {msg}")
def alert(msg):   print(f"  {Fore.RED}🚨{Style.RESET_ALL}  {msg}")
def info(msg):    print(f"  {Fore.CYAN}ℹ{Style.RESET_ALL}  {msg}")
def note(msg):    print(f"  {Fore.MAGENTA}→{Style.RESET_ALL}  {msg}")

def section(title, char="─", width=70):
    print(f"\n{Fore.CYAN}{char*width}")
    print(f"  {Fore.WHITE}{title}")
    print(f"{Fore.CYAN}{char*width}{Style.RESET_ALL}")

def fmt_diff(val):
    """Colorir diferença percentual."""
    if val >= 500:  return f"{Fore.RED}{val:>8.1f}%{Style.RESET_ALL}"
    if val >= 100:  return f"{Fore.YELLOW}{val:>8.1f}%{Style.RESET_ALL}"
    if val >= 30:   return f"{Fore.CYAN}{val:>8.1f}%{Style.RESET_ALL}"
    return f"{Style.DIM}{val:>8.1f}%{Style.RESET_ALL}"

def fmt_score(s):
    if s >= 0.75: return f"{Fore.GREEN}{s:.3f}{Style.RESET_ALL}"
    if s >= 0.45: return f"{Fore.YELLOW}{s:.3f}{Style.RESET_ALL}"
    return f"{Fore.RED}{s:.3f}{Style.RESET_ALL}"


# ─── Carregamento ──────────────────────────────────────────────────────────────
def detect_sep(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
    for sep in [",", ";", "\t", "|"]:
        if sep in sample:
            return sep
    return ","

def load_files(path: str, label_col: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Carrega um ou vários CSVs, separa normal vs ataque.
    Retorna (df_normal, df_attack, nomes_ficheiros_carregados).
    """
    p = Path(path)
    files = [p] if p.is_file() else sorted(p.glob("*.csv"))
    if not files:
        print(f"{Fore.RED}Nenhum CSV encontrado em {path}{Style.RESET_ALL}")
        sys.exit(1)

    section(f"A carregar {len(files)} ficheiro(s)")
    all_normal, all_attack, loaded_names = [], [], []

    for f in files:
        try:
            sep = detect_sep(f)
            df  = pd.read_csv(f, sep=sep, encoding="utf-8", low_memory=False)
            rows = len(df)

            # Determinar se é normal ou ataque
            if label_col and label_col in df.columns:
                # Usar coluna de label para separar
                n = df[df[label_col] == 0].drop(columns=[label_col], errors="ignore")
                a = df[df[label_col] == 1].drop(columns=[label_col], errors="ignore")
                label_src = f"coluna '{label_col}'"
            else:
                # Inferir pelo nome do ficheiro
                fname = f.name.lower()
                if any(kw in fname for kw in ["normal", "benign", "legit"]):
                    n, a = df, pd.DataFrame()
                    label_src = "nome do ficheiro (normal)"
                elif any(kw in fname for kw in ["attack", "brute", "scan", "flood",
                                                  "dos", "mitm", "inject", "malform"]):
                    n, a = pd.DataFrame(), df
                    label_src = "nome do ficheiro (ataque)"
                else:
                    warn(f"{f.name} — não foi possível determinar tipo, a ignorar.")
                    continue

            n_norm, n_att = len(n), len(a)
            ok(f"{f.name:<45} {rows:>7} linhas  "
               f"[normal={n_norm:>6} / ataque={n_att:>6}]  via {label_src}")
            all_normal.append(n)
            all_attack.append(a)
            loaded_names.append(f.name)
        except Exception as e:
            warn(f"{f.name} — erro: {e}")

    df_normal = pd.concat([d for d in all_normal if not d.empty], ignore_index=True)
    df_attack  = pd.concat([d for d in all_attack  if not d.empty], ignore_index=True)

    info(f"Total normal: {len(df_normal):,} fluxos  |  Total ataque: {len(df_attack):,} fluxos")
    return df_normal, df_attack, loaded_names


# ─── Features relevantes ───────────────────────────────────────────────────────
def get_features(df_normal, df_attack, label_col) -> list[str]:
    """Colunas numéricas presentes em ambos, excluindo identificadores."""
    skip = NON_FEATURES | {label_col}
    num_n = set(df_normal.select_dtypes(include=np.number).columns) - skip
    num_a = set(df_attack.select_dtypes(include=np.number).columns) - skip
    return sorted(num_n & num_a)


# ─── Análise estatística ───────────────────────────────────────────────────────
def compute_comparison(df_n: pd.DataFrame, df_a: pd.DataFrame,
                        features: list[str]) -> pd.DataFrame:
    """
    Para cada feature calcula estatísticas dos dois grupos
    e métricas de discriminação.
    """
    rows = []
    for f in features:
        n_vals = df_n[f].dropna()
        a_vals = df_a[f].dropna()

        if len(n_vals) == 0 or len(a_vals) == 0:
            continue

        n_mean, n_med, n_std = n_vals.mean(), n_vals.median(), n_vals.std()
        a_mean, a_med, a_std = a_vals.mean(), a_vals.median(), a_vals.std()

        # Diferença percentual das médias
        denom   = (abs(n_mean) + abs(a_mean)) / 2 + 1e-12
        diff_pct = abs(n_mean - a_mean) / denom * 100

        # Cohen's d (effect size)
        pooled_std = np.sqrt((n_std**2 + a_std**2) / 2 + 1e-12)
        cohens_d   = abs(n_mean - a_mean) / pooled_std

        # Mann-Whitney U (non-parametric, robusto)
        try:
            u_stat, p_val = scipy_stats.mannwhitneyu(
                n_vals.sample(min(len(n_vals), 5000), random_state=42),
                a_vals.sample(min(len(a_vals), 5000), random_state=42),
                alternative="two-sided"
            )
            significant = p_val < 0.05
        except Exception:
            p_val, significant = 1.0, False

        rows.append({
            "feature":       f,
            "normal_mean":   round(n_mean, 4),
            "normal_median": round(n_med, 4),
            "normal_std":    round(n_std, 4),
            "attack_mean":   round(a_mean, 4),
            "attack_median": round(a_med, 4),
            "attack_std":    round(a_std, 4),
            "diff_%":        round(diff_pct, 1),
            "cohens_d":      round(cohens_d, 3),
            "p_value":       round(p_val, 6),
            "significativo": significant,
            "n_normal":      len(n_vals),
            "n_attack":      len(a_vals),
        })

    df = pd.DataFrame(rows)
    return df


# ─── Score de discriminação ────────────────────────────────────────────────────
def compute_scores(comp: pd.DataFrame) -> pd.DataFrame:
    """
    Score composto de discriminação (0-1):
      - 50%: Cohen's d normalizado
      - 30%: diferença % normalizada
      - 20%: significância estatística
    """
    df = comp.copy()

    # Normalizar Cohen's d (cap em 10)
    cd_norm = (df["cohens_d"].clip(0, 10) / 10)

    # Normalizar diff_%  (cap em 1000%)
    dp_norm = (df["diff_%"].clip(0, 1000) / 1000)

    # Significância
    sig = df["significativo"].astype(float)

    df["score"] = (cd_norm * 0.50 + dp_norm * 0.30 + sig * 0.20).round(4)

    # Recomendação
    def recommend(row):
        if row["score"] >= 0.60: return "★★★ Excelente"
        if row["score"] >= 0.40: return "★★☆ Boa"
        if row["score"] >= 0.20: return "★☆☆ Moderada"
        return "☆☆☆ Fraca"

    df["recomendação"] = df.apply(recommend, axis=1)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ─── Display ───────────────────────────────────────────────────────────────────
def display_comparison_table(df_scores: pd.DataFrame, threshold: float):
    section("Comparação Normal vs Ataque — Todas as Features")
    cols = ["feature", "normal_mean", "attack_mean", "diff_%",
            "normal_std", "attack_std", "cohens_d", "p_value"]
    display = df_scores[cols].copy()

    # Imprimir linha a linha com cor na diff_%
    header = f"  {'Feature':<25} {'Norm.Mean':>12} {'Att.Mean':>12} {'Diff%':>9}  {'Norm.Std':>10} {'Att.Std':>10} {'Cohen d':>8}  {'p-value':>9}"
    print(f"{Fore.WHITE}{header}{Style.RESET_ALL}")
    print("  " + "─"*100)

    for _, row in display.iterrows():
        diff = row["diff_%"]
        if diff >= threshold:
            diff_str = f"{Fore.RED}{diff:>8.1f}%{Style.RESET_ALL}"
        elif diff >= threshold * 0.5:
            diff_str = f"{Fore.YELLOW}{diff:>8.1f}%{Style.RESET_ALL}"
        else:
            diff_str = f"{Style.DIM}{diff:>8.1f}%{Style.RESET_ALL}"

        print(f"  {row['feature']:<25} {row['normal_mean']:>12.3f} {row['attack_mean']:>12.3f} {diff_str}  "
              f"{row['normal_std']:>10.3f} {row['attack_std']:>10.3f} {row['cohens_d']:>8.3f}  {row['p_value']:>9.5f}")


def display_alerts(df_scores: pd.DataFrame, threshold: float):
    section("🚨 Alertas — Features com Maior Diferença Normal vs Ataque")
    high = df_scores[df_scores["diff_%"] >= threshold].copy()
    if high.empty:
        info(f"Nenhuma feature com diferença > {threshold}%")
        return

    for _, row in high.iterrows():
        n_m, a_m = row["normal_mean"], row["attack_mean"]
        direction = "↑ maior em ataques" if a_m > n_m else "↓ menor em ataques"
        alert(
            f"{row['feature']:<28}  diff={row['diff_%']:>7.1f}%  "
            f"Cohen's d={row['cohens_d']:>6.2f}  {direction}"
        )
        note(f"    Normal: mean={n_m:.3f}  median={row['normal_median']:.3f}  std={row['normal_std']:.3f}")
        note(f"    Ataque: mean={a_m:.3f}  median={row['attack_median']:.3f}  std={row['attack_std']:.3f}")


def display_groups(df_scores: pd.DataFrame):
    section("Análise por Grupo de Features")
    known = set()
    for group, cols in FEATURE_GROUPS.items():
        present = [c for c in cols if c in df_scores["feature"].values]
        if not present:
            continue
        known.update(present)
        print(f"\n  {Fore.YELLOW}▶ {group}{Style.RESET_ALL}")
        sub = df_scores[df_scores["feature"].isin(present)][
            ["feature", "normal_mean", "attack_mean", "diff_%", "cohens_d", "score", "recomendação"]
        ]
        print(tabulate(sub, headers="keys", tablefmt="simple", floatfmt=".3f",
                       showindex=False, colalign=("left",)))

    # Features fora dos grupos
    unknown = [f for f in df_scores["feature"] if f not in known]
    if unknown:
        print(f"\n  {Fore.YELLOW}▶ Outras Features{Style.RESET_ALL}")
        sub = df_scores[df_scores["feature"].isin(unknown)][
            ["feature", "normal_mean", "attack_mean", "diff_%", "cohens_d", "score", "recomendação"]
        ]
        print(tabulate(sub, headers="keys", tablefmt="simple", floatfmt=".3f",
                       showindex=False, colalign=("left",)))


def display_ranking(df_scores: pd.DataFrame, top_n: int):
    section(f"🏆 Ranking de Features para Treino ML (Top {top_n})")
    top = df_scores.head(top_n)[["feature","score","diff_%","cohens_d",
                                   "normal_mean","attack_mean","recomendação"]].copy()
    # Imprimir com cor no score
    print(f"  {'#':>3}  {'Feature':<28} {'Score':>6}  {'Diff%':>8}  "
          f"{'Cohen d':>8}  {'Norm.Mean':>10}  {'Att.Mean':>10}  Recom.")
    print("  " + "─"*100)
    for i, (_, row) in enumerate(top.iterrows(), 1):
        score_str = fmt_score(row["score"])
        rec_color = (Fore.GREEN if "★★★" in row["recomendação"] else
                     Fore.YELLOW if "★★☆" in row["recomendação"] else
                     Fore.RED)
        print(f"  {i:>3}.  {row['feature']:<28} {score_str}  "
              f"{row['diff_%']:>7.1f}%  {row['cohens_d']:>8.3f}  "
              f"{row['normal_mean']:>10.3f}  {row['attack_mean']:>10.3f}  "
              f"{rec_color}{row['recomendação']}{Style.RESET_ALL}")

    # Resumo final
    best = df_scores[df_scores["recomendação"].str.contains("★★★")]["feature"].tolist()
    good = df_scores[df_scores["recomendação"].str.contains("★★☆")]["feature"].tolist()
    section("Resumo para Uso em ML", char="═")
    if best:
        print(f"\n  {Fore.GREEN}Features excelentes (usar sempre):{Style.RESET_ALL}")
        print(f"    {', '.join(best)}")
    if good:
        print(f"\n  {Fore.YELLOW}Features boas (considerar):{Style.RESET_ALL}")
        print(f"    {', '.join(good)}")
    note("Estas features apresentam maior separação estatística entre tráfego")
    note("normal e ataques — ideais para Random Forest, Isolation Forest, SVM, etc.")


# ─── Exportação ────────────────────────────────────────────────────────────────
def export(df_scores: pd.DataFrame, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_scores.to_csv(out / "feature_analysis.csv", index=False)
    ok(f"Análise completa    → {out / 'feature_analysis.csv'}")

    best = df_scores[df_scores["recomendação"].str.contains("★★")]
    best[["feature","score","diff_%","cohens_d","recomendação"]].to_csv(
        out / "features_recomendadas.csv", index=False)
    ok(f"Features recomend.  → {out / 'features_recomendadas.csv'}")

    # Gerar snippet Python para uso directo em ML
    # Usar score numérico directamente (evita falhas com comparação de strings Unicode)
    # Thresholds adaptativos: se nada passa nos limiares fixos, usar Top-N ou percentil
    score_col = df_scores["score"]
    
    excelentes = df_scores[score_col >= 0.60]["feature"].tolist()
    boas        = df_scores[(score_col >= 0.40) & (score_col < 0.60)]["feature"].tolist()
    moderadas   = df_scores[(score_col >= 0.20) & (score_col < 0.40)]["feature"].tolist()

    # Fallback adaptativo: se os limiares fixos não apanharem nada útil,
    # usar o percentil 70 do score como limiar mínimo
    if not excelentes and not boas:
        p70 = float(score_col.quantile(0.70))
        excelentes = df_scores[score_col >= p70]["feature"].tolist()
        boas        = []
        moderadas   = df_scores[(score_col >= score_col.quantile(0.40)) &
                                (score_col < p70)]["feature"].tolist()
        adaptive_note = f"# NOTA: Limiares fixos não produziram resultados — usado percentil 70 do score ({p70:.3f}) como limiar adaptativo.\n"
    else:
        adaptive_note = ""

    feat_ml = excelentes + boas
    if not feat_ml:
        feat_ml = moderadas  # último recurso: incluir moderadas

    snippet = f"""# Features seleccionadas automaticamente pelo IoT IDS Feature Analyzer
# Score baseado em Cohen's d + diferença % + significância estatística (Mann-Whitney U)
{adaptive_note}
FEATURES_EXCELENTES = {excelentes}
FEATURES_BOAS       = {boas}
FEATURES_MODERADAS  = {moderadas}
FEATURES_ML         = FEATURES_EXCELENTES + FEATURES_BOAS  # recomendado para treino

# Se FEATURES_ML estiver vazio (dataset com baixa separação), usar:
# FEATURES_ML = FEATURES_EXCELENTES + FEATURES_BOAS + FEATURES_MODERADAS

# Uso:
# X = df[FEATURES_ML]
# y = df['is_attack']
# clf = RandomForestClassifier().fit(X, y)
# clf = IsolationForest().fit(X)
"""
    with open(out / "features_ml.py", "w") as fh:
        fh.write(snippet)
    ok(f"Snippet Python ML   → {out / 'features_ml.py'}")


# ─── Modo interactivo ──────────────────────────────────────────────────────────
def interactive_mode(df_normal, df_attack, df_scores):
    section("Modo Interactivo")
    cmds = [
        ("feature <nome>",   "Distribuição detalhada de uma feature"),
        ("top <n>",          "Top N features no ranking"),
        ("group <nome>",     "Features de um grupo (IAT, Bytes, Pkts, Flags, Len)"),
        ("export <pasta>",   "Exportar todos os resultados"),
        ("quit",             "Sair"),
    ]
    for cmd, desc in cmds:
        print(f"    {Fore.CYAN}{cmd:<26}{Style.RESET_ALL} {desc}")

    group_map = {
        "iat":   "Inter-Arrival Time (IAT)",
        "bytes": "Bytes",
        "pkts":  "Volumes de Pacotes",
        "flags": "Flags TCP",
        "len":   "Comprimento de Pacotes",
    }

    while True:
        try:
            raw = input(f"\n{Fore.GREEN}iot-analyzer>{Style.RESET_ALL} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nA sair...")
            break

        if not raw:
            continue
        parts = raw.split()
        cmd   = parts[0].lower()

        if cmd in ("quit", "exit", "q"):
            print("Até logo!")
            break

        elif cmd == "feature" and len(parts) >= 2:
            feat = " ".join(parts[1:])
            if feat not in df_scores["feature"].values:
                warn(f"Feature '{feat}' não encontrada.")
                continue
            row = df_scores[df_scores["feature"] == feat].iloc[0]
            section(f"Detalhe: {feat}")
            n_vals = df_normal[feat].dropna()
            a_vals = df_attack[feat].dropna()
            tbl = [
                ["", "Normal", "Ataque"],
                ["Contagem",  f"{len(n_vals):,}",       f"{len(a_vals):,}"],
                ["Média",     f"{n_vals.mean():.4f}",   f"{a_vals.mean():.4f}"],
                ["Mediana",   f"{n_vals.median():.4f}", f"{a_vals.median():.4f}"],
                ["Std Dev",   f"{n_vals.std():.4f}",    f"{a_vals.std():.4f}"],
                ["Mínimo",    f"{n_vals.min():.4f}",    f"{a_vals.min():.4f}"],
                ["Máximo",    f"{n_vals.max():.4f}",    f"{a_vals.max():.4f}"],
                ["P25",       f"{n_vals.quantile(.25):.4f}", f"{a_vals.quantile(.25):.4f}"],
                ["P75",       f"{n_vals.quantile(.75):.4f}", f"{a_vals.quantile(.75):.4f}"],
            ]
            print(tabulate(tbl, headers="firstrow", tablefmt="rounded_outline"))
            print(f"\n  Score: {fmt_score(row['score'])}   "
                  f"Diff%: {row['diff_%']:.1f}%   "
                  f"Cohen's d: {row['cohens_d']:.3f}   "
                  f"{row['recomendação']}")

        elif cmd == "top":
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
            display_ranking(df_scores, top_n=n)

        elif cmd == "group" and len(parts) >= 2:
            key = parts[1].lower()
            gname = group_map.get(key)
            if not gname:
                warn(f"Grupo desconhecido. Usa: {', '.join(group_map.keys())}")
                continue
            cols = FEATURE_GROUPS.get(gname, [])
            sub  = df_scores[df_scores["feature"].isin(cols)]
            if sub.empty:
                warn("Nenhuma feature deste grupo encontrada no dataset.")
            else:
                section(f"Grupo: {gname}")
                print(tabulate(
                    sub[["feature","normal_mean","attack_mean","diff_%",
                          "cohens_d","score","recomendação"]],
                    headers="keys", tablefmt="rounded_outline",
                    floatfmt=".4f", showindex=False
                ))

        elif cmd == "export":
            folder = parts[1] if len(parts) > 1 else "./resultados_iot"
            export(df_scores, folder)

        else:
            warn(f"Comando desconhecido: '{raw}'")


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Análise de features IoT IDS — Normal vs Ataque"
    )
    parser.add_argument("--path",      default=".",
                        help="Pasta com CSVs ou caminho para um ficheiro CSV")
    parser.add_argument("--label",     default="is_attack",
                        help="Nome da coluna de label (0=normal, 1=ataque). "
                             "Se ausente, o tipo é inferido pelo nome do ficheiro.")
    parser.add_argument("--threshold", type=float, default=50.0,
                        help="Limiar %% de diferença para emitir alerta (default: 50)")
    parser.add_argument("--top",       type=int,   default=15,
                        help="Número de features no ranking final (default: 15)")
    parser.add_argument("--export",    default=None,
                        help="Pasta para exportar resultados (opcional)")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Não entrar em modo interactivo")
    args = parser.parse_args()

    print(BANNER)

    # 1. Carregar dados
    df_normal, df_attack, _ = load_files(args.path, args.label)

    if df_normal.empty:
        print(f"{Fore.RED}Sem dados normais carregados.{Style.RESET_ALL}")
        sys.exit(1)
    if df_attack.empty:
        print(f"{Fore.RED}Sem dados de ataque carregados.{Style.RESET_ALL}")
        sys.exit(1)

    # 2. Features relevantes
    features = get_features(df_normal, df_attack, args.label)
    section(f"Features Numéricas Relevantes: {len(features)}")
    info(f"Excluídas automaticamente: {', '.join(sorted(NON_FEATURES & set(df_normal.columns)))}")
    info(f"A analisar: {', '.join(features)}")

    if not features:
        print(f"{Fore.RED}Nenhuma feature numérica comum encontrada.{Style.RESET_ALL}")
        sys.exit(1)

    # 3. Comparação estatística
    section("A calcular estatísticas comparativas...")
    comp      = compute_comparison(df_normal, df_attack, features)
    df_scores = compute_scores(comp)

    # 4. Display
    display_comparison_table(df_scores, args.threshold)
    display_alerts(df_scores, args.threshold)
    display_groups(df_scores)
    display_ranking(df_scores, args.top)

    # 5. Exportação automática
    if args.export:
        section("A exportar resultados")
        export(df_scores, args.export)

    # 6. Modo interactivo
    if not args.no_interactive:
        interactive_mode(df_normal, df_attack, df_scores)


if __name__ == "__main__":
    main()