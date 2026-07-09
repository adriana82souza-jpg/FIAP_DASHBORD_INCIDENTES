"""
LocaVisionAI — Dashboard de Análise de Incidentes
PredictOps Team | FIAP Challenge Sprint 3

Execute com:
    streamlit run LocaVisionAI/src/dashboard_v2.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from PIL import Image
from plotly.subplots import make_subplots
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────
_icon = Image.open(Path(__file__).parent.parent / "Imagens" / "logo.png")
_gra  = Image.open(Path(__file__).parent.parent / "Imagens" / "gra.png")
_engre = Image.open(Path(__file__).parent.parent / "Imagens" / "engre.png")

st.set_page_config(
    page_title="LocaVisionAI — Incidentes",
    page_icon=_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ESTILO CUSTOMIZADO
# ─────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetricValue"] { 
        font-size: 2.5rem;
        font-weight: 900;    
    }
    .block-container { padding-top: 1.5rem; }
    h1 { color: #1f3d7a; }
    h2 { color: #2c5f9e; border-bottom: 2px solid #e0e8f0; padding-bottom: 0.3rem; }
    h3 { color: #3a7abf; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CARREGAMENTO DE DADOS
# ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_preparation import carregar_df_preparado  # noqa: E402


@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    df = carregar_df_preparado()
    # Aliases para compatibilidade com o restante do dashboard
    df["DiaSemana"]   = df["dia_semana_nome"]
    df["Hora"]        = df["hora_abertura"]
    df["Ano"]         = df["ano"]
    df["is_kpi_base"] = df["eh_incidente_pai"].astype(bool)
    df["Mes_Abertura"] = df["Mes_Abertura"].astype(str)
    return df


df_full = load_data()

# ─────────────────────────────────────────────
# SIDEBAR — FILTROS
# ─────────────────────────────────────────────
with st.sidebar:
    st.image(_icon, width=200)
    st.markdown("## LocaVisionAI")
    st.caption("PredictOps — Sprint 3")
    st.divider()

    st.image(_gra, width=30)
    st.markdown("### Filtros")

    # Período
    min_date = df_full["Aberto"].min().date()
    max_date = df_full["Aberto"].max().date()
    date_range = st.date_input(
        "Período de abertura",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Prioridade
    prioridades = sorted(df_full["Prioridade"].dropna().unique())
    sel_prioridade = st.multiselect(
        "Prioridade",
        options=prioridades,
        default=prioridades,
    )

    # Status
    statuses = sorted(df_full["Status"].dropna().unique())
    sel_status = st.multiselect(
        "Status",
        options=statuses,
        default=statuses,
    )

    # Grupo designado
    grupos = sorted(df_full["Grupo designado"].dropna().unique())
    sel_grupo = st.multiselect(
        "Grupo designado",
        options=grupos,
        default=grupos,
    )

    # Produto (top 20 + outros)
    top_produtos = df_full["Produto"].value_counts().head(20).index.tolist()
    sel_produto = st.multiselect(
        "Produto (top 20)",
        options=top_produtos,
        default=[],
        placeholder="Todos",
    )

    st.divider()
    TOP_N = st.slider("Top N itens nos gráficos", 5, 30, 15)
    st.divider()
    st.caption("Dados: df_prepared.pkl")

# ─────────────────────────────────────────────
# APLICAR FILTROS
# ─────────────────────────────────────────────
try:
    if len(date_range) == 2:
        start_date, end_date = date_range[0], date_range[1]
        df = df_full[
            (df_full["Aberto"].dt.date >= start_date)
            & (df_full["Aberto"].dt.date <= end_date)
        ].copy()
    else:
        df = df_full.copy()
except Exception:
    df = df_full.copy()

if sel_prioridade:
    df = df[df["Prioridade"].isin(sel_prioridade)]
if sel_status:
    df = df[df["Status"].isin(sel_status)]
if sel_grupo:
    df = df[df["Grupo designado"].isin(sel_grupo)]
if sel_produto:
    df = df[df["Produto"].isin(sel_produto)]

df_kpi = df[df["eh_incidente_pai"] == 1].copy()

# ─────────────────────────────────────────────
# CABEÇALHO
# ─────────────────────────────────────────────
col_logo, col_titulo = st.columns([1, 8])
with col_logo:
    st.image(_icon, width=150)
with col_titulo:
    st.title("LocaVisionAI — Análise de Incidentes")
st.caption(f"Exibindo **{len(df):,}** incidentes com os filtros aplicados  |  Base total: {len(df_full):,}")
st.divider()

# ─────────────────────────────────────────────
# PARTE 1 — MÉTRICAS GERAIS (KPI CARDS)
# ─────────────────────────────────────────────
st.header("📊 Visão Geral")

col1, col2, col3, col4, col5 = st.columns(5)

total = len(df)
p1_p2 = len(df[df["Prioridade"].isin(["1 - Crítica", "2 - Alta"])])
pct_p1p2 = (p1_p2 / total * 100) if total > 0 else 0

kpi_base_total = len(df_kpi[df_kpi["entrou_kpi_flag"] == 1])
kpi_violado = len(df_kpi[df_kpi["KPI Violado?"] == "SIM"])
pct_kpi = (kpi_violado / kpi_base_total * 100) if kpi_base_total > 0 else 0

dur_media = df["Duração"].median() / 60 if "Duração" in df.columns else 0  # em minutos → horas

with col1:
    st.metric("Total de Incidentes", f"{total:,}".replace(",", "."))
with col2:
    st.metric("P1 + P2 (Críticos)", f"{p1_p2:,}".replace(",", "."), f"{pct_p1p2:.1f}%".replace(".", ","))
with col3:
    st.metric("Base KPI Elegível", f"{kpi_base_total:,}".replace(",", "."))
with col4:
    st.metric("KPI Violado", f"{kpi_violado:,}".replace(",", "."), f"{pct_kpi:.1f}%".replace(".", ","), delta_color="inverse")
with col5:
    st.metric("Duração Mediana", f"{dur_media:.1f}h".replace(".", ","))

st.divider()

# ─────────────────────────────────────────────
# PARTE 1.0 — MÉTRICAS GERAIS (Gráfico evolutivo)
# ─────────────────────────────────────────────
vol_mes = (
    df.groupby("Mes_Abertura")
    .size()
    .reset_index(name="Incidentes")
    .sort_values("Mes_Abertura")
)

fig_vol = px.line(
    vol_mes,
    x="Mes_Abertura",
    y="Incidentes",
    title="Volume Mensal de Incidentes",
    markers=True,
    text="Incidentes", 
    color_discrete_sequence=["#2c5f9e"],
)

# 1. Ajusta espessura, tamanho dos marcadores e ABREVIA os rótulos
fig_vol.update_traces(
    line=dict(width=3),
    marker=dict(size=10),
    textposition="top center",
    texttemplate="%{y:.2s}"  # <--- ADICIONADO: Abrevia os números (ex: 25000 vira 25k, 3500 vira 3.5k)
)

# 2. Ajusta o layout (Filtra a data e formata o eixo)
fig_vol.update_layout(
    height=450,
    xaxis=dict(
        title="Mês Abertura",
        tickangle=320,
        dtick="M1",
        tickformat="%b %Y",
        range=["2023-01-01", "2025-12-31"] # <--- ADICIONADO: Trava o gráfico entre Jan/2023 e Dez/2025
    ),
    yaxis=dict(
        dtick=5000,
        range=[0, 30000],
        title="Quantidade"
    ),
    margin=dict(t=40, b=40)
)

# Renderiza o gráfico dentro da col_d
st.plotly_chart(fig_vol, use_container_width=True, key="grafico_volume_principal")

# ─────────────────────────────────────────────
# PARTE 2 — DISTRIBUIÇÃO POR PRIORIDADE E STATUS
# ─────────────────────────────────────────────

st.header("🏷️ Prioridade e Status")

col_a, col_b , col_c = st.columns(3)

with col_a:
    status_counts = df["Prioridade"].value_counts().reset_index()
    status_counts.columns = ["Prioridade", "Qtd"]
    # Usamos cores fixas para cada prioridade
    cores_prioridade = {
        "1 - Crítica": "#d32f2f",      # vermelho
        "2 - Alta": "#d63676",         # Azul
        "3 - Média": "#c14773",        # Azul escuro
        "4 - Baixa": "#09034E",        # Azul claro
        "5 - Muito Baixa": "#080141"   # Azul mais claro
    }
    
    fig_prioridade = px.bar(
        status_counts,
        x="Prioridade",
        y="Qtd",
        orientation="v",
        title="Distribuição por Prioridade",
        color="Prioridade",           # Alterado para usar cores por categoria
        color_discrete_map=cores_prioridade, # Aplica as cores fixas
        category_orders={"Prioridade": ["1 - Crítica", "2 - Alta", "3 - Média", "4 - Baixa", "5 - Muito Baixa"]}
        #text="Qtd", #remove o rotulo de dados
    )
    
    # Ajustes de visualização das barras
    #fig_prioridade.update_traces(
    #    textposition="outside", 
    #    cliponaxis=False
    #)
    
    fig_prioridade.update_layout(
        height=500,
        showlegend=False,         # Remove a legenda lateral
        xaxis=dict(
            tickangle=320,         # Ajusta para 320 graus (quase vertical)
        ),
        yaxis=dict(
            dtick=10000,          # Escala de 10 em 10 mil
            range=[0, 70000],     # Limite em 70 mil
            title="Quantidade"
        ),
        xaxis_title="",
        margin=dict(t=80, b=150)  # Aumentei um pouco o 'b' para os nomes em 90º
    )
    
    st.plotly_chart(fig_prioridade, use_container_width=True)

with col_b:
    pri_counts = df["Prioridade"].value_counts().reset_index()
    pri_counts.columns = ["Prioridade", "Qtd"]

    #st.write("Dados para o gráfico:", pri_counts)

    # 1. Definindo cores exatas para cada nível de prioridade
    cores_prioridade = {
        "1 - Crítica": "#d32f2f",      # vermelho
        "2 - Alta": "#d63676",         # Azul
        "3 - Média": "#c14773",        # Azul escuro
        "4 - Baixa": "#09034E",        # Azul claro
        "5 - Muito Baixa": "#080141"   # Azul mais claro
    }

    fig_pri = px.pie(
        pri_counts,
        names="Prioridade",
        values="Qtd",
        color="Prioridade",
        title="Distribuição por Prioridade",
        color_discrete_map=cores_prioridade,
        hole=0.4,
    )
    fig_pri.update_traces(textposition="inside", textinfo="percent")
    fig_pri.update_layout(height=450, showlegend=True, legend=dict(orientation="h", y=-0.15),
    margin=dict(t=80, b=60, l=50, r=50))
    #fig_pri.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig_pri, use_container_width=True)

with col_c:
    status_counts = df["Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Qtd"]
    # Usamos cores fixas para cada status
    cores_status = {
        "Sem Intervenção": "#d63676",
        "Encerrado Automaticamente": "#09034E",
        "Encerrado": "#09034E",
        "Aguardando Problema": "#09034E"
    }
    
    fig_status = px.bar(
        status_counts,
        x="Status",
        y="Qtd",
        orientation="v",
        title="Distribuição por Status",
        color="Status",           # Alterado para usar cores por categoria
        color_discrete_map=cores_status, # Aplica as cores fixas
        #text="Qtd",
    )
    
    # Ajustes de visualização das barras
    #fig_status.update_traces(
    #    textposition="outside", 
    #    cliponaxis=False
    #)
    
    fig_status.update_layout(
        height=500,
        showlegend=False,         # Remove a legenda lateral
        xaxis=dict(
        tickangle=320,         # Ajusta para 320 graus (quase vertical)
        ),
        yaxis=dict(
            dtick=10000,          # Escala de 10 em 10 mil
            range=[0, 90000],     # Limite em 90 mil
            title="Quantidade"
        ),
        xaxis_title="",
        margin=dict(t=80, b=150)  # Aumentei um pouco o 'b' para os nomes em 90º
    )
    
    st.plotly_chart(fig_status, use_container_width=True)

# ─────────────────────────────────────────────
# PARTE 3 — ANÁLISE TEMPORAL
# ─────────────────────────────────────────────
st.header("📅 Análise Temporal")

# Criamos as duas colunas
col_d, col_e = st.columns(2)

# --- GRÁFICO DA ESQUERDA ---
with col_d:
    # 1. Defina a ordem exata que você quer (ajuste os nomes se na sua base tiver "-feira")
    ordem_dias = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    
    # 2. Transforma a coluna em uma categoria com a ordem que criamos acima
    df["dia_semana_pt"] = pd.Categorical(df["dia_semana_pt"], categories=ordem_dias, ordered=True)
    
    # 3. Agora o sort_index() vai respeitar a nossa lista, e não o alfabeto!
    dia_counts = df["dia_semana_pt"].value_counts().sort_index().reset_index()
    dia_counts.columns = ["Dia da Semana", "Qtd"]
    
    fig_dia = px.bar(
        dia_counts,
        x="Dia da Semana",
        y="Qtd",
        title="Abertura por Dia da Semana",
        color="Qtd",
        color_continuous_scale=["#d63676","#c14773", "#09034E", "#09034E"], #alterei a cor para um gradiente de azul invertido
        #color_discrete_sequence=["#1f77b4"], # <--- ADICIONE ESTA LINHA COM A SUA COR
    )
    
    fig_dia.update_layout(
        height=450,                   
        coloraxis_showscale=False,
        xaxis=dict(                   
            title="Dia da Semana"     
        ),
        yaxis=dict(
            dtick=1000,               
            range=[0, 20000],          
            title="Quantidade"        
        ),
        margin=dict(t=40, b=40)
    )    
    
    # Renderiza o gráfico dentro da col_d
    st.plotly_chart(fig_dia, use_container_width=True)

# --- GRÁFICO DA DIREITA ---
with col_e:
    # 1. Filtra a base para manter APENAS as prioridades 1, 2 e 3
    df_prio = df[df["Prioridade_Num"].isin([1, 2, 3])].copy()

    # 2. Define a ordem exata que você quer
    ordem_dias = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    
    # 3. Transforma a coluna em uma categoria com a ordem na base FILTRADA
    df_prio["dia_semana_pt"] = pd.Categorical(df_prio["dia_semana_pt"], categories=ordem_dias, ordered=True)
    
    # 4. Faz a contagem respeitando a nossa lista
    dia_counts = df_prio["dia_semana_pt"].value_counts().sort_index().reset_index()
    dia_counts.columns = ["Dia da Semana", "Qtd"]
    
    # 5. Gráfico vertical exato que você pediu
    fig_dia = px.bar(
        dia_counts,
        x="Dia da Semana",
        y="Qtd",
        title="Abertura por Dia da Semana (P1, P2 e P3)",
        color="Qtd",
        color_continuous_scale=["#d63676","#c14773", "#09034E", "#09034E"], 
    )
    
    fig_dia.update_layout(
        height=450,                   
        coloraxis_showscale=False,
        xaxis=dict(                   
            title="Dia da Semana"     
        ),
        yaxis=dict(
            dtick=1000,               
            range=[0, 11000],          
            title="Quantidade"        
        ),
        margin=dict(t=40, b=40)
    )    
    
    # Renderiza o gráfico dentro da col_e
    st.plotly_chart(fig_dia, use_container_width=True)

# Divisor na tela toda após todos os gráficos
st.divider()

# Criamos as duas colunas

col_f, col_g = st.columns(2)

with col_f:
     # 1. Filtra a base para manter APENAS as prioridades 1, 2 e 3
    hora_counts = df["hora_abertura"].value_counts().sort_index().reset_index()
    hora_counts.columns = ["Hora", "Qtd"]
    
    fig_hora = px.bar(
        hora_counts,
        x="Hora",
        y="Qtd",
        title="Abertura por Hora do Dia",
        color="Qtd",
       color_continuous_scale=
                        ["#d63676","#d63676","#d63676","#c14773","#c14773",
                         "#09034E","#09034E", "#09034E","#09034E", "#09034E"
                        ], 
    )
    
    fig_hora.update_layout(
        height=450,                   # Mesma altura do gráfico ao lado para alinhar
        coloraxis_showscale=False,
        xaxis=dict(                   # <--- ADICIONADO: Configuração do eixo X
            dtick=1,                  # Força a exibição de 1 em 1
            title="Hora do Dia"       # Nome do eixo (opcional)
        ),
        yaxis=dict(
            dtick=1000,               # Escala de 1 em 1 mil
            range=[0, 7000],          # Limite em 7 mil
            title="Quantidade"        # Corrigida a indentação
        ),
        margin=dict(t=40, b=40)
    )    
  
    # Renderiza o gráfico dentro da col_f
    st.plotly_chart(fig_hora, use_container_width=True)
    
with col_g:
    df_prio = df[df["Prioridade_Num"].isin([1, 2, 3])].copy()
    hora_counts = df_prio["hora_abertura"].value_counts().sort_index().reset_index()
    hora_counts.columns = ["Hora", "Qtd"]
    
    fig_hora = px.bar(
        hora_counts,
        x="Hora",
        y="Qtd",
        title="Abertura por Hora do Dia (P1, P2 e P3)",
        color="Qtd",
       color_continuous_scale=
                        ["#d63676","#d63676","#d63676","#c14773","#c14773",
                         "#09034E","#09034E", "#09034E","#09034E", "#09034E"
                        ], 
    )
    
    fig_hora.update_layout(
        height=450,                   # Mesma altura do gráfico ao lado para alinhar
        coloraxis_showscale=False,
        xaxis=dict(                   # <--- ADICIONADO: Configuração do eixo X
            dtick=1,                  # Força a exibição de 1 em 1
            title="Hora do Dia"       # Nome do eixo (opcional)
        ),
        yaxis=dict(
            dtick=1000,               # Escala de 1 em 1 mil
            range=[0, 4500],          # Limite em 4,5 mil
            title="Quantidade"        # Corrigida a indentação
        ),
        margin=dict(t=40, b=40)
    )    
  
    # Renderiza o gráfico dentro da col_g
    st.plotly_chart(fig_hora, use_container_width=True)    

# --- GRÁFICO INFERIOR (HEATMAP) ---
# Fora do bloco "with", para ocupar a largura total da página abaixo dos gráficos de barras
heatmap_data = pd.crosstab(df["dia_semana_pt"], df["hora_abertura"])

fig_heat = px.imshow(
    heatmap_data,
    labels=dict(x="Hora do Dia", y="Dia da Semana", color="Qtd Incidentes"),
    x=heatmap_data.columns,
    y=heatmap_data.index,
    color_continuous_scale="YlOrRd",
    aspect="auto",
    title="Heatmap de Sazonalidade — Dia da Semana × Hora do Dia"
)

fig_heat.update_traces(
    xgap=2,
    ygap=2
)

fig_heat.update_layout(
    height=500,
    xaxis=dict(
        dtick=1,
        side="bottom"
    ),
    margin=dict(t=50, b=40, l=40, r=40),
    coloraxis_colorbar=dict(
        title=dict(
            text="Qtd Incidentes",
            side="right"           # <--- Agora sim o Plotly entende que é o lado do título!
        ),
        thickness=15,       
        len=1.04,           
        y=0.5,              
        yanchor="middle"
    )
)
st.plotly_chart(fig_heat, use_container_width=True)


# Divisor na tela toda após todos os gráficos
st.divider()

# ─────────────────────────────────────────────
# PARTE 4 — KPI E QUALIDADE DE SERVIÇO
# ─────────────────────────────────────────────
st.header("🎯 KPI e Qualidade de Serviço")

col_f, col_g = st.columns(2)

with col_f:
    kpi_counts = df_kpi["KPI Violado?"].value_counts().reset_index()
    kpi_counts.columns = ["KPI Violado?", "Qtd"]
    fig_kpi = px.pie(
        kpi_counts,
        names="KPI Violado?",
        values="Qtd",
        title="KPI Violado (base elegível)",
        color_discrete_map={"SIM": "#e74c3c", "NAO": "#2ecc71", "N/A": "#bdc3c7"},
        hole=0.4,
    )
    fig_kpi.update_traces(textposition="inside", textinfo="percent+label")
    fig_kpi.update_layout(legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig_kpi, use_container_width=True)

with col_g:
    entrou_counts = df_kpi["Entrou para KPI?"].value_counts().reset_index()
    entrou_counts.columns = ["Entrou para KPI?", "Qtd"]
    fig_entrou = px.bar(
        entrou_counts,
        x="Entrou para KPI?",
        y="Qtd",
        title="Entrou para KPI?",
        color="Entrou para KPI?",
        color_discrete_map={"SIM": "#3498db", "NAO": "#95a5a6"},
        text="Qtd",
    )
    fig_entrou.update_traces(textposition="outside")
    fig_entrou.update_layout(showlegend=False)
    st.plotly_chart(fig_entrou, use_container_width=True)

# Taxa de violação por prioridade
kpi_pri = (
    df_kpi[df_kpi["KPI Violado?"].isin(["SIM", "NAO"])]
    .groupby("Prioridade")["KPI Violado?"]
    .apply(lambda x: (x == "SIM").mean() * 100)
    .reset_index(name="Taxa Violação (%)")
    .sort_values("Taxa Violação (%)", ascending=True)
)
fig_kpi_pri = px.bar(
    kpi_pri,
    x="Taxa Violação (%)",
    y="Prioridade",
    orientation="h",
    title="Taxa de Violação de KPI por Prioridade",
    color="Taxa Violação (%)",
    color_continuous_scale="RdYlGn_r",
    text="Taxa Violação (%)",
)
fig_kpi_pri.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig_kpi_pri.update_layout(coloraxis_showscale=False)
st.plotly_chart(fig_kpi_pri, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# PARTE 5 — PRODUTO / CATEGORIA / GRUPO
# ─────────────────────────────────────────────
st.header("📦 Produto, Categoria e Grupo Designado")

tab1, tab2, tab3 = st.tabs(["🏭 Produto", "🗂️ Categoria", "👥 Grupo Designado"])

with tab1:
    col_g, col_h = st.columns(2)
    top_prod = df["Produto"].value_counts().head(TOP_N).reset_index()
    top_prod.columns = ["Produto", "Qtd"]

    with col_g:
        fig_prod = px.bar(
            top_prod.sort_values("Qtd"),
            x="Qtd",
            y="Produto",
            orientation="h",
            title=f"Top {TOP_N} Produtos por Volume",
            color="Qtd",
            color_continuous_scale="Blues",
        )
        fig_prod.update_layout(coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_prod, use_container_width=True)

    with col_h:
        kpi_prod = (
            df_kpi[df_kpi["KPI Violado?"].isin(["SIM", "NAO"])]
            .groupby("Produto")["KPI Violado?"]
            .apply(lambda x: (x == "SIM").mean() * 100)
            .reset_index(name="Taxa (%)")
            .nlargest(TOP_N, "Taxa (%)")
            .sort_values("Taxa (%)")
        )
        fig_kpi_prod = px.bar(
            kpi_prod,
            x="Taxa (%)",
            y="Produto",
            orientation="h",
            title=f"Top {TOP_N} Produtos — Taxa KPI Violado",
            color="Taxa (%)",
            color_continuous_scale="RdYlGn_r",
            text="Taxa (%)",
        )
        fig_kpi_prod.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_kpi_prod.update_layout(coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_kpi_prod, use_container_width=True)

with tab2:
    col_i, col_j = st.columns(2)
    top_cat = df["Categoria"].value_counts().head(TOP_N).reset_index()
    top_cat.columns = ["Categoria", "Qtd"]

    with col_i:
        fig_cat = px.bar(
            top_cat.sort_values("Qtd"),
            x="Qtd",
            y="Categoria",
            orientation="h",
            title=f"Top {TOP_N} Categorias por Volume",
            color="Qtd",
            color_continuous_scale="Oranges",
        )
        fig_cat.update_layout(coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_j:
        kpi_cat = (
            df_kpi[df_kpi["KPI Violado?"].isin(["SIM", "NAO"])]
            .groupby("Categoria")["KPI Violado?"]
            .apply(lambda x: (x == "SIM").mean() * 100)
            .reset_index(name="Taxa (%)")
            .nlargest(TOP_N, "Taxa (%)")
            .sort_values("Taxa (%)")
        )
        fig_kpi_cat = px.bar(
            kpi_cat,
            x="Taxa (%)",
            y="Categoria",
            orientation="h",
            title=f"Top {TOP_N} Categorias — Taxa KPI Violado",
            color="Taxa (%)",
            color_continuous_scale="RdYlGn_r",
            text="Taxa (%)",
        )
        fig_kpi_cat.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_kpi_cat.update_layout(coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_kpi_cat, use_container_width=True)

with tab3:
    col_k, col_l = st.columns(2)
    top_grp = df["Grupo designado"].value_counts().head(TOP_N).reset_index()
    top_grp.columns = ["Grupo", "Qtd"]

    with col_k:
        fig_grp = px.bar(
            top_grp.sort_values("Qtd"),
            x="Qtd",
            y="Grupo",
            orientation="h",
            title=f"Top {TOP_N} Grupos por Volume",
            color="Qtd",
            color_continuous_scale="Greens",
        )
        fig_grp.update_layout(coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_grp, use_container_width=True)

    with col_l:
        # Distribuição de prioridade por grupo (stacked)
        grp_pri = (
            df.groupby(["Grupo designado", "Prioridade"])
            .size()
            .reset_index(name="Qtd")
        )
        top_grupos = df["Grupo designado"].value_counts().head(TOP_N).index
        grp_pri = grp_pri[grp_pri["Grupo designado"].isin(top_grupos)]
        fig_grp_pri = px.bar(
            grp_pri,
            x="Qtd",
            y="Grupo designado",
            color="Prioridade",
            orientation="h",
            title=f"Distribuição de Prioridade por Grupo (top {TOP_N})",
            barmode="stack",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_grp_pri.update_layout(yaxis=dict(categoryorder="total ascending"), yaxis_title="")
        st.plotly_chart(fig_grp_pri, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# PARTE 6 — TABELA DETALHADA
# ─────────────────────────────────────────────
st.header("🗃️ Dados Detalhados")

with st.expander("Exibir tabela filtrada", expanded=False):
    cols_show = [
        "Número", "Prioridade", "Status", "Produto", "Categoria",
        "Grupo designado", "Aberto", "Encerrado", "Duração",
        "KPI Violado?", "Entrou para KPI?",
    ]
    cols_available = [c for c in cols_show if c in df.columns]
    st.dataframe(
        df[cols_available].sort_values("Aberto", ascending=False),
        use_container_width=True,
        height=400,
    )

    csv = df[cols_available].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Baixar CSV filtrado",
        data=csv,
        file_name="incidentes_filtrado.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────
# RODAPÉ
# ─────────────────────────────────────────────
st.divider()
st.caption("LocaVisionAI · PredictOps Team · FIAP Challenge Sprint 3 · 2025")
