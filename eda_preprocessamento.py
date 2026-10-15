"""
Projeto 2ª VA - Inteligência Artificial (UFRPE / UABJ) - Entrega 1
Definição, Análise Exploratória e Pré-processamento do Dataset

Dataset: Netflix Customer Churn (Kaggle)
Tarefa:  classificação binária - prever se o cliente cancelou a assinatura (coluna `churned`)

Como executar (na pasta do projeto):
    pip install -r requirements.txt
    python eda_preprocessamento.py

Saídas:
    figuras/                          -> gráficos da análise exploratória
    data/processed/netflix_churn_encoded.csv -> dados codificados (sem escala)
    data/processed/netflix_churn_scaled.csv  -> dados codificados + padronizados
    relatorio_eda.txt                 -> resumo textual de toda a análise
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera as imagens sem abrir janelas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 0. Configurações e caminhos
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
ARQ_ORIGINAL = BASE / "data" / "original" / "netflix_customer_churn.csv"
DIR_PROCESSADO = BASE / "data" / "processed"
DIR_FIGURAS = BASE / "figuras"
ARQ_RELATORIO = BASE / "relatorio_eda.txt"

DIR_PROCESSADO.mkdir(parents=True, exist_ok=True)
DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

ALVO = "churned"
sns.set_theme(style="whitegrid")

# Tudo o que é impresso também vai para o relatório em texto
linhas_relatorio = []


def log(texto=""):
    print(texto)
    linhas_relatorio.append(str(texto))


def titulo(texto):
    log()
    log("=" * 70)
    log(texto)
    log("=" * 70)


# ---------------------------------------------------------------------------
# 1. Definição do conjunto de dados
# ---------------------------------------------------------------------------
df = pd.read_csv(ARQ_ORIGINAL)

titulo("1. DEFINIÇÃO DO CONJUNTO DE DADOS")
log("Fonte: Kaggle - Netflix Customer Churn")
log("Domínio: clientes de um serviço de streaming (dados de perfil, uso e assinatura).")
log(f"Atributo-alvo: '{ALVO}' (1 = cliente cancelou, 0 = cliente permaneceu).")
log("Tarefa: classificação binária.")

# ---------------------------------------------------------------------------
# 2. Análise detalhada: exemplos, atributos, tipos e valores ausentes
# ---------------------------------------------------------------------------
titulo("2. ESTRUTURA DO DATASET")
log(f"Número de exemplos (linhas): {df.shape[0]}")
log(f"Número de colunas: {df.shape[1]} (1 identificador + {df.shape[1] - 2} atributos preditores + 1 alvo)")

# Classificação manual de cada coluna quanto ao seu tipo semântico
numericas = ["age", "watch_hours", "last_login_days", "monthly_fee",
             "number_of_profiles", "avg_watch_time_per_day"]
categoricas = ["gender", "subscription_type", "region", "device",
               "payment_method", "favorite_genre"]

tipos = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "valores_unicos": df.nunique(),
    "ausentes": df.isna().sum(),
    "ausentes_%": (df.isna().mean() * 100).round(2),
})
tipos["tipo_semantico"] = [
    "identificador" if c == "customer_id"
    else "alvo (binário)" if c == ALVO
    else "numérico" if c in numericas
    else "categórico"
    for c in df.columns
]
log(tipos.to_string())

log()
log(f"Total de valores ausentes: {df.isna().sum().sum()}")
log(f"Linhas duplicadas: {df.duplicated().sum()}")
log(f"IDs de cliente duplicados: {df['customer_id'].duplicated().sum()}")

# ---------------------------------------------------------------------------
# 3. Distribuição dos atributos
# ---------------------------------------------------------------------------
titulo("3. DISTRIBUIÇÃO DOS ATRIBUTOS")

log("Estatísticas descritivas dos atributos numéricos:")
desc = df[numericas].describe().T
desc["assimetria"] = df[numericas].skew()
log(desc.round(3).to_string())

log()
log("Frequência das categorias de cada atributo categórico:")
for col in categoricas:
    log(f"\n{col}:")
    log(df[col].value_counts().to_string())

# Histogramas dos atributos numéricos
fig, eixos = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(eixos.ravel(), numericas):
    sns.histplot(df[col], kde=True, ax=ax, color="#4C72B0")
    ax.set_title(f"{col} (assimetria = {df[col].skew():.2f})")
fig.suptitle("Distribuição dos atributos numéricos")
fig.tight_layout()
fig.savefig(DIR_FIGURAS / "01_histogramas_numericos.png", dpi=120)
plt.close(fig)

# Boxplots para visualizar outliers
fig, eixos = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(eixos.ravel(), numericas):
    sns.boxplot(x=df[col], ax=ax, color="#DD8452")
    ax.set_title(col)
fig.suptitle("Boxplots dos atributos numéricos (detecção de outliers)")
fig.tight_layout()
fig.savefig(DIR_FIGURAS / "02_boxplots_numericos.png", dpi=120)
plt.close(fig)

# Gráficos de barras dos atributos categóricos
fig, eixos = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(eixos.ravel(), categoricas):
    sns.countplot(data=df, x=col, ax=ax, order=df[col].value_counts().index, color="#55A868")
    ax.set_title(col)
    ax.tick_params(axis="x", rotation=35)
fig.suptitle("Distribuição dos atributos categóricos")
fig.tight_layout()
fig.savefig(DIR_FIGURAS / "03_barras_categoricos.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Balanceamento das classes
# ---------------------------------------------------------------------------
titulo("4. BALANCEAMENTO DAS CLASSES")
contagem = df[ALVO].value_counts().sort_index()
proporcao = df[ALVO].value_counts(normalize=True).sort_index() * 100
for classe in contagem.index:
    log(f"Classe {classe}: {contagem[classe]} exemplos ({proporcao[classe]:.1f}%)")
razao = contagem.max() / contagem.min()
log(f"Razão maioria/minoria: {razao:.3f}")
log("Conclusão: classes praticamente balanceadas -> não é necessário "
    "oversampling/undersampling (SMOTE etc.).")

fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(data=df, x=ALVO, ax=ax, hue=ALVO, palette=["#4C72B0", "#C44E52"], legend=False)
ax.set_xticks([0, 1], ["0 - permaneceu", "1 - cancelou"])
ax.set_title("Distribuição da classe-alvo")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
fig.tight_layout()
fig.savefig(DIR_FIGURAS / "04_balanceamento_classes.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Relação dos atributos com o alvo e entre si
# ---------------------------------------------------------------------------
titulo("5. RELAÇÃO DOS ATRIBUTOS COM O ALVO")

# Correlação de Pearson entre numéricos (inclui o alvo binário)
corr = df[numericas + [ALVO]].corr()
log("Correlação de cada atributo numérico com o alvo:")
log(corr[ALVO].drop(ALVO).sort_values().round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Matriz de correlação (atributos numéricos + alvo)")
fig.tight_layout()
fig.savefig(DIR_FIGURAS / "05_matriz_correlacao.png", dpi=120)
plt.close(fig)

# Distribuição dos numéricos separada por classe
fig, eixos = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(eixos.ravel(), numericas):
    sns.boxplot(data=df, x=ALVO, y=col, ax=ax, hue=ALVO,
                palette=["#4C72B0", "#C44E52"], legend=False)
    ax.set_title(f"{col} por classe")
fig.suptitle("Atributos numéricos x churn")
fig.tight_layout()
fig.savefig(DIR_FIGURAS / "06_numericos_por_classe.png", dpi=120)
plt.close(fig)

# Taxa de churn por categoria
log()
log("Taxa de churn (%) por categoria:")
fig, eixos = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(eixos.ravel(), categoricas):
    taxa = (df.groupby(col)[ALVO].mean() * 100).sort_values(ascending=False)
    log(f"\n{col}:")
    log(taxa.round(1).to_string())
    sns.barplot(x=taxa.index, y=taxa.values, ax=ax, color="#8172B2")
    ax.axhline(df[ALVO].mean() * 100, ls="--", color="black", lw=1)
    ax.set_ylabel("% churn")
    ax.set_title(col)
    ax.tick_params(axis="x", rotation=35)
fig.suptitle("Taxa de churn por categoria (linha tracejada = média geral)")
fig.tight_layout()
fig.savefig(DIR_FIGURAS / "07_churn_por_categoria.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. Investigação de redundâncias e outliers
# ---------------------------------------------------------------------------
titulo("6. REDUNDÂNCIAS E OUTLIERS")

# 6.1 monthly_fee x subscription_type: cada plano tem um único preço?
precos = df.groupby("subscription_type")["monthly_fee"].unique()
log("Valores de monthly_fee por plano:")
log(precos.to_string())
log("-> monthly_fee é totalmente determinado por subscription_type (atributo redundante).")

# 6.2 avg_watch_time_per_day parece ser derivado de watch_hours e last_login_days
estimado = df["watch_hours"] / (df["last_login_days"] + 1)
erro = (df["avg_watch_time_per_day"] - estimado).abs()
log()
log("Hipótese: avg_watch_time_per_day ≈ watch_hours / (last_login_days + 1)")
log(f"Erro absoluto médio da hipótese: {erro.mean():.4f} (máx. {erro.max():.4f})")
log("-> atributo derivado; valores extremos ocorrem quando last_login_days = 0.")

# 6.3 Outliers pelo critério do IQR (1,5 x intervalo interquartil)
log()
log("Outliers pelo critério IQR (1,5 x IQR):")
q1, q3 = df[numericas].quantile(0.25), df[numericas].quantile(0.75)
iqr = q3 - q1
mascara_outlier = (df[numericas] < q1 - 1.5 * iqr) | (df[numericas] > q3 + 1.5 * iqr)
outliers = pd.DataFrame({
    "qtd_outliers": mascara_outlier.sum(),
    "%": (mascara_outlier.mean() * 100).round(2),
})
log(outliers.to_string())
log("-> Outliers apenas em watch_hours e avg_watch_time_per_day. São valores plausíveis "
    "(clientes que assistem muito), então NÃO serão removidos; a assimetria será "
    "reduzida com transformação log(1 + x).")

# ---------------------------------------------------------------------------
# 7. Pré-processamento
# ---------------------------------------------------------------------------
titulo("7. PRÉ-PROCESSAMENTO APLICADO")
proc = df.copy()

# 7.1 Remoção do identificador: não carrega informação preditiva
proc = proc.drop(columns=["customer_id"])
log("[1] customer_id removido (identificador único, sem valor preditivo).")

# 7.2 Remoção de monthly_fee: redundante com subscription_type
proc = proc.drop(columns=["monthly_fee"])
log("[2] monthly_fee removido (redundante com subscription_type).")

# 7.3 Transformação log nos atributos muito assimétricos (trata outliers sem perder linhas)
for col in ["watch_hours", "avg_watch_time_per_day"]:
    proc[col] = np.log1p(proc[col])
log("[3] log(1 + x) aplicado em watch_hours e avg_watch_time_per_day.")
log(f"    Assimetria após log: watch_hours = {proc['watch_hours'].skew():.2f}, "
    f"avg_watch_time_per_day = {proc['avg_watch_time_per_day'].skew():.2f}")

fig, eixos = plt.subplots(1, 2, figsize=(12, 4))
for ax, col in zip(eixos, ["watch_hours", "avg_watch_time_per_day"]):
    sns.histplot(proc[col], kde=True, ax=ax, color="#4C72B0")
    ax.set_title(f"log1p({col}) (assimetria = {proc[col].skew():.2f})")
fig.tight_layout()
fig.savefig(DIR_FIGURAS / "08_apos_transformacao_log.png", dpi=120)
plt.close(fig)

# 7.4 Codificação ordinal: subscription_type tem ordem natural (Basic < Standard < Premium)
ordem_planos = {"Basic": 0, "Standard": 1, "Premium": 2}
proc["subscription_type"] = proc["subscription_type"].map(ordem_planos)
log("[4] subscription_type codificado como ordinal: Basic=0, Standard=1, Premium=2.")

# 7.5 One-hot encoding nos categóricos nominais (sem ordem entre as categorias)
nominais = ["gender", "region", "device", "payment_method", "favorite_genre"]
proc = pd.get_dummies(proc, columns=nominais, dtype=int)
log(f"[5] One-hot encoding em {nominais}.")

# Coloca o alvo como última coluna
proc = proc[[c for c in proc.columns if c != ALVO] + [ALVO]]
proc.to_csv(DIR_PROCESSADO / "netflix_churn_encoded.csv", index=False)
log(f"    -> data/processed/netflix_churn_encoded.csv: {proc.shape[0]} linhas x {proc.shape[1]} colunas")

# 7.6 Padronização (média 0, desvio 1) dos atributos contínuos/ordinais.
#     Importante para algoritmos baseados em distância/gradiente (KNN, SVM, MLP, Regressão Logística).
#     OBS: na Entrega 2, o StandardScaler deve ser ajustado APENAS no conjunto de treino
#     (dentro de um Pipeline) para evitar vazamento de dados; esta versão serve para análise.
continuas = ["age", "watch_hours", "last_login_days", "number_of_profiles",
             "avg_watch_time_per_day", "subscription_type"]
escalado = proc.copy()
escalado[continuas] = StandardScaler().fit_transform(escalado[continuas])
escalado.to_csv(DIR_PROCESSADO / "netflix_churn_scaled.csv", index=False)
log(f"[6] StandardScaler aplicado em {continuas}.")
log(f"    -> data/processed/netflix_churn_scaled.csv: {escalado.shape[0]} linhas x {escalado.shape[1]} colunas")

log()
log("Itens que NÃO precisaram de tratamento:")
log("  - Valores ausentes: nenhum encontrado.")
log("  - Duplicatas: nenhuma encontrada.")
log("  - Balanceamento: classes ~50/50, nenhuma técnica de reamostragem necessária.")

titulo("RESUMO FINAL")
log(f"Original:        {df.shape[0]} exemplos x {df.shape[1]} colunas")
log(f"Pré-processado:  {proc.shape[0]} exemplos x {proc.shape[1] - 1} atributos + 1 alvo")
log(f"Figuras salvas em: {DIR_FIGURAS.relative_to(BASE)}/")

ARQ_RELATORIO.write_text("\n".join(linhas_relatorio), encoding="utf-8")
