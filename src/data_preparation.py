"""
LocaVision AI — Sprint 3
data_preparation.py

Pipeline único de carregamento, limpeza e feature engineering.
Consolida as transformações de 01_exploratory_analysis, 02_preprocessing
e 03_model_training em um único módulo reutilizável.

Saídas geradas em ../data/:
  - df_prepared.pkl      → DataFrame completo, tratado e enriquecido
  - df_kpi_base.pkl      → Apenas incidentes PAI (elegíveis para KPI)
  - train_test_split.pkl → Splits prontos para classificação de prioridade
  - ts_<label>.pkl       → Séries temporais diárias (Total, P2-Alta, P3-Média)
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CAMINHOS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "data")
CSV_PATH  = os.path.join(DATA_DIR, "LW-DATASET.csv")

# Features para o modelo de classificação de prioridade
FEATURES_CLASSIFICACAO = [
    "hora_abertura",
    "dia_semana_num",
    "mes",
    "eh_fim_semana",
    "eh_horario_comercial",
    "duracao_min",
    "kpi_violado_flag",
    "entrou_kpi_flag",
    "tipo_problema_enc",
    "status_enc",
    "ic_enc",
]

TARGET_CLASSIFICACAO = "Prioridade_Num"

# Features para o modelo de série temporal (volume diário)
FEATURES_TS = [
    "dia_semana",
    "mes",
    "dia_mes",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "rolling_7",
    "rolling_30",
]

# Top N itens de configuração a codificar individualmente
TOP_N_IC = 20


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARREGAMENTO
# ─────────────────────────────────────────────────────────────────────────────
def carregar_raw(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Lê o CSV bruto."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"[1] Raw carregado: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. LIMPEZA BÁSICA
# ─────────────────────────────────────────────────────────────────────────────
def limpar(df: pd.DataFrame) -> pd.DataFrame:
    """Converte tipos, remove registros sem data de abertura."""
    df = df.copy()

    # Datas
    df["Aberto"]    = pd.to_datetime(df["Aberto"],    errors="coerce")
    df["Encerrado"] = pd.to_datetime(df["Encerrado"], errors="coerce")

    # Remove registros sem data de abertura (não utilizáveis)
    antes = len(df)
    df = df.dropna(subset=["Aberto"]).reset_index(drop=True)
    print(f"[2] Removidos {antes - len(df):,} registros sem data de abertura. "
          f"Restam: {len(df):,}")

    # Duração em segundos → minutos
    df["Duração"]    = pd.to_numeric(df["Duração"], errors="coerce")
    df["duracao_min"] = df["Duração"] / 60

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURES TEMPORAIS
# ─────────────────────────────────────────────────────────────────────────────
def criar_features_temporais(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva colunas de data/hora a partir de 'Aberto'."""
    df = df.copy()

    df["hora_abertura"]       = df["Aberto"].dt.hour
    df["dia_semana_num"]      = df["Aberto"].dt.dayofweek        # 0=Seg, 6=Dom
    df["dia_semana_nome"]     = df["Aberto"].dt.day_name()       # Ex: Monday
    df["dia_mes"]             = df["Aberto"].dt.day
    df["mes"]                 = df["Aberto"].dt.month
    df["ano"]                 = df["Aberto"].dt.year
    df["Mes_Abertura"]        = df["Aberto"].dt.to_period("M")
    df["Periodo"]             = df["Aberto"].dt.to_period("M")
    df["eh_fim_semana"]       = (df["dia_semana_num"] >= 5).astype(int)
    df["eh_horario_comercial"] = df["hora_abertura"].between(8, 18).astype(int)

    # Mapa para nome do dia em PT-BR (útil em gráficos)
    _mapa_dia_pt = {
        "Monday": "Seg", "Tuesday": "Ter", "Wednesday": "Qua",
        "Thursday": "Qui", "Friday": "Sex", "Saturday": "Sáb", "Sunday": "Dom",
    }
    df["dia_semana_pt"] = df["dia_semana_nome"].map(_mapa_dia_pt)

    print("[3] Features temporais criadas.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURES DE PRIORIDADE
# ─────────────────────────────────────────────────────────────────────────────
def criar_features_prioridade(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai número da prioridade e cria colunas derivadas."""
    df = df.copy()

    # Número da prioridade: '3 - Média' → 3
    df["prio_num"]       = df["Prioridade"].str.extract(r"(\d+)").astype(float)
    df["Prioridade_Num"] = df["prio_num"]   # alias legível

    # Flag P1/P2 (incidentes críticos)
    df["eh_critico"] = (df["prio_num"] <= 2).astype(int)

    print("[4] Features de prioridade criadas.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURES DE KPI / OLA
# ─────────────────────────────────────────────────────────────────────────────
def criar_features_kpi(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas de KPI para flags numéricas."""
    df = df.copy()

    df["kpi_violado_flag"] = (
        df["KPI Violado?"].fillna("NÃO").str.upper() == "SIM"
    ).astype(int)

    df["entrou_kpi_flag"] = (
        df["Entrou para KPI?"].fillna("NÃO").str.upper() == "SIM"
    ).astype(int)

    # Flag incidente PAI (sem referência de Incidente Pai → elegível para KPI)
    df["eh_incidente_pai"] = df["Incidente Pai"].isna().astype(int)

    print("[5] Features de KPI criadas.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. TIPO DE PROBLEMA (NLP básico)
# ─────────────────────────────────────────────────────────────────────────────
def _extrair_tipo_problema(descricao) -> str:
    if pd.isna(descricao):
        return "Desconhecido"
    d = str(descricao).lower()
    mapa = [
        ("apache",   "Apache"),
        ("timeout",  "Timeout"),
        ("disk",     "Disco"),
        ("space",    "Disco"),
        ("memory",   "Memória"),
        ("mem ",     "Memória"),
        ("cpu",      "CPU"),
        ("network",  "Rede"),
        ("net ",     "Rede"),
        ("database", "Banco de Dados"),
        (" db ",     "Banco de Dados"),
        ("backup",   "Backup"),
    ]
    for kw, label in mapa:
        if kw in d:
            return label
    return "Outro"


def criar_feature_tipo_problema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tipo_problema"] = df["Descrição resumida"].apply(_extrair_tipo_problema)
    print("[6] Feature tipo_problema criada.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS
# ─────────────────────────────────────────────────────────────────────────────
def codificar_categoricas(
    df: pd.DataFrame,
    top_n_ic: int = TOP_N_IC,
) -> tuple[pd.DataFrame, dict]:
    """
    Aplica LabelEncoder nas variáveis categóricas relevantes.
    Retorna o DataFrame enriquecido e um dicionário com os encoders
    treinados (para uso em inferência posterior).
    """
    df = df.copy()
    encoders = {}

    # Tipo de problema
    le_tipo = LabelEncoder()
    df["tipo_problema_enc"] = le_tipo.fit_transform(df["tipo_problema"])
    encoders["tipo_problema"] = le_tipo

    # Status
    le_status = LabelEncoder()
    df["Status_clean"] = df["Status"].fillna("Desconhecido")
    df["status_enc"]   = le_status.fit_transform(df["Status_clean"])
    encoders["status"] = le_status

    # Item de Configuração — top N mais frequentes; resto → "Outro"
    top_ics = df["Item de configuração"].value_counts().head(top_n_ic).index.tolist()
    df["ic_clean"] = df["Item de configuração"].apply(
        lambda x: str(x) if x in top_ics else "Outro"
    )
    le_ic = LabelEncoder()
    df["ic_enc"] = le_ic.fit_transform(df["ic_clean"].fillna("Outro"))
    encoders["ic"] = le_ic

    print(f"[7] Encoders ajustados: tipo_problema ({len(le_tipo.classes_)} classes), "
          f"status ({len(le_status.classes_)} classes), "
          f"ic ({len(le_ic.classes_)} classes)")
    return df, encoders


# ─────────────────────────────────────────────────────────────────────────────
# 8. SUBSET: INCIDENTES PAI (elegíveis para KPI)
# ─────────────────────────────────────────────────────────────────────────────
def criar_df_kpi_base(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra apenas incidentes PAI (sem referência a Incidente Pai)."""
    df_kpi = df[df["Incidente Pai"].isna()].copy()
    print(f"[8] df_kpi_base: {len(df_kpi):,} incidentes PAI "
          f"({len(df_kpi)/len(df)*100:.1f}% do total)")
    return df_kpi


# ─────────────────────────────────────────────────────────────────────────────
# 9. DIVISÃO TREINO / TESTE — CLASSIFICAÇÃO DE PRIORIDADE
# ─────────────────────────────────────────────────────────────────────────────
def criar_split_classificacao(
    df: pd.DataFrame,
    features: list = FEATURES_CLASSIFICACAO,
    target: str = TARGET_CLASSIFICACAO,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Prepara e divide o dataset para o modelo de classificação de prioridade.
    Remove registros com NaN nas features/target e classes com < 2 amostras.
    """
    df_model = df[features + [target]].copy().dropna()

    # Remove classes com menos de 2 amostras (impossibilita stratify)
    contagem = df_model[target].value_counts()
    classes_validas = contagem[contagem >= 2].index
    removidas = sorted(set(contagem.index) - set(classes_validas))
    if removidas:
        print(f"[9] Classes removidas (< 2 amostras): {removidas}")
    df_model = df_model[df_model[target].isin(classes_validas)].copy()

    X = df_model[features]
    y = df_model[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[9] Split criado: treino {X_train.shape} | teste {X_test.shape} | "
          f"classes: {sorted(y.unique())}")

    return {
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "features": features,
        "target":   target,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 10. SÉRIES TEMPORAIS DIÁRIAS (para previsão de volume D+1 / D+7)
# ─────────────────────────────────────────────────────────────────────────────
def _criar_features_ts(serie: pd.Series) -> pd.DataFrame:
    """
    Converte uma série temporal de volume diário em DataFrame com
    features de lag e rolling para o RandomForestRegressor.
    """
    dfx = pd.DataFrame({
        "Data":   pd.to_datetime(serie.index),
        "Volume": serie.values,
    }).sort_values("Data").reset_index(drop=True)

    dfx["dia_semana"] = dfx["Data"].dt.dayofweek
    dfx["mes"]        = dfx["Data"].dt.month
    dfx["dia_mes"]    = dfx["Data"].dt.day

    for lag in [1, 2, 3, 7, 14]:
        dfx[f"lag_{lag}"] = dfx["Volume"].shift(lag)

    dfx["rolling_7"]  = dfx["Volume"].rolling(7).mean().shift(1)
    dfx["rolling_30"] = dfx["Volume"].rolling(30).mean().shift(1)

    return dfx.dropna().reset_index(drop=True)


def criar_series_temporais(df: pd.DataFrame) -> dict:
    """
    Gera séries temporais diárias para os três segmentos exigidos:
      - Total
      - P2 - Alta
      - P3 - Média

    Retorna um dicionário label → DataFrame com features de TS prontas.
    """
    filtros = {
        "Total":      df,
        "P2 - Alta":  df[df["Prioridade"].str.startswith("2", na=False)],
        "P3 - Média": df[df["Prioridade"].str.startswith("3", na=False)],
    }

    series = {}
    for label, subset in filtros.items():
        serie = subset.groupby(subset["Aberto"].dt.date).size()
        ts_df = _criar_features_ts(serie)
        series[label] = ts_df
        print(f"[10] Série TS '{label}': {len(ts_df):,} dias | "
              f"volume médio: {ts_df['Volume'].mean():.1f}")

    return series


# ─────────────────────────────────────────────────────────────────────────────
# 11. SALVAMENTO
# ─────────────────────────────────────────────────────────────────────────────
def salvar_artefatos(
    df: pd.DataFrame,
    df_kpi: pd.DataFrame,
    split: dict,
    series_ts: dict,
    encoders: dict,
    data_dir: str = DATA_DIR,
) -> None:
    """Persiste todos os artefatos gerados em disco."""
    os.makedirs(data_dir, exist_ok=True)

    # DataFrame principal
    df.to_pickle(os.path.join(data_dir, "df_prepared.pkl"))
    print(f"[11] Salvo: df_prepared.pkl  ({df.shape[0]:,} × {df.shape[1]})")

    # Subset KPI base
    df_kpi.to_pickle(os.path.join(data_dir, "df_kpi_base.pkl"))
    print(f"[11] Salvo: df_kpi_base.pkl  ({df_kpi.shape[0]:,} linhas)")

    # Split de classificação
    with open(os.path.join(data_dir, "train_test_split.pkl"), "wb") as f:
        pickle.dump(split, f)
    print(f"[11] Salvo: train_test_split.pkl")

    # Séries temporais
    for label, ts_df in series_ts.items():
        fname = "ts_" + label.lower().replace(" ", "_").replace("-", "").replace("__", "_") + ".pkl"
        ts_df.to_pickle(os.path.join(data_dir, fname))
        print(f"[11] Salvo: {fname}")

    # Encoders (para inferência)
    with open(os.path.join(data_dir, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    print(f"[11] Salvo: encoders.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def executar_pipeline(
    csv_path: str = CSV_PATH,
    data_dir: str = DATA_DIR,
    salvar: bool = True,
) -> dict:
    """
    Executa o pipeline completo de preparação de dados.

    Retorna um dicionário com todos os artefatos gerados:
      - df          : DataFrame completo e enriquecido
      - df_kpi_base : Subset incidentes PAI
      - split       : Dicionário com X_train, X_test, y_train, y_test, features, target
      - series_ts   : Dicionário com DataFrames de série temporal por segmento
      - encoders    : Dicionário com LabelEncoders treinados

    Uso:
        from src.data_preparation import executar_pipeline
        artefatos = executar_pipeline()
        df = artefatos['df']
    """
    print("=" * 60)
    print("LocaVision AI — Pipeline de Preparação de Dados")
    print("=" * 60)

    df = carregar_raw(csv_path)
    df = limpar(df)
    df = criar_features_temporais(df)
    df = criar_features_prioridade(df)
    df = criar_features_kpi(df)
    df = criar_feature_tipo_problema(df)
    df, encoders = codificar_categoricas(df)

    df_kpi  = criar_df_kpi_base(df)
    split   = criar_split_classificacao(df)
    series_ts = criar_series_temporais(df)

    if salvar:
        salvar_artefatos(df, df_kpi, split, series_ts, encoders, data_dir)

    print("=" * 60)
    print("Pipeline concluído com sucesso.")
    print(f"  DataFrame final : {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    print(f"  Novas colunas   : {[c for c in df.columns if c not in pd.read_csv(csv_path, nrows=0, encoding='utf-8').columns]}")
    print("=" * 60)

    return {
        "df":          df,
        "df_kpi_base": df_kpi,
        "split":       split,
        "series_ts":   series_ts,
        "encoders":    encoders,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UTILITÁRIOS DE CARREGAMENTO (para uso nos dashboards/notebooks)
# ─────────────────────────────────────────────────────────────────────────────
def carregar_df_preparado(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Carrega o DataFrame tratado salvo em disco. Executa o pipeline se não existir."""
    path = os.path.join(data_dir, "df_prepared.pkl")
    if not os.path.exists(path):
        print("df_prepared.pkl não encontrado. Executando pipeline...")
        executar_pipeline(data_dir=data_dir)
    return pd.read_pickle(path)


def carregar_split(data_dir: str = DATA_DIR) -> dict:
    """Carrega o split de treino/teste. Executa o pipeline se não existir."""
    path = os.path.join(data_dir, "train_test_split.pkl")
    if not os.path.exists(path):
        executar_pipeline(data_dir=data_dir)
    with open(path, "rb") as f:
        return pickle.load(f)


def carregar_series_ts(data_dir: str = DATA_DIR) -> dict:
    """Carrega as séries temporais. Executa o pipeline se não existir."""
    labels = {"Total": "ts_total.pkl",
              "P2 - Alta": "ts_p2_alta.pkl",
              "P3 - Média": "ts_p3_m\u00e9dia.pkl"}
    series = {}
    for label, fname in labels.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            executar_pipeline(data_dir=data_dir)
        series[label] = pd.read_pickle(path)
    return series


# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO DIRETA
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    executar_pipeline()
