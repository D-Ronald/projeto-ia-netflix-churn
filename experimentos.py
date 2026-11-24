"""
Projeto 2ª VA - Inteligência Artificial (UFRPE/UABJ) - Entrega 2
Comparação de algoritmos de classificação no dataset Netflix Customer Churn.

Procedimento experimental
-------------------------
* Algoritmos (scikit-learn): Árvore de Decisão, KNN, Naive Bayes (Gaussiano),
  Regressão Logística, SVM e Rede Neural MLP.
* Validação: 10-fold cross-validation ESTRATIFICADO (mesmas 10 partições para
  todos os algoritmos, garantindo comparação pareada).
* Para cada algoritmo, uma grade de hiperparâmetros (>= 3 combinações) é avaliada
  com GridSearchCV. Para CADA combinação são coletadas, em CADA fold, as métricas
  acurácia, precisão, recall, F1 e AUC-ROC; ao final calcula-se média e desvio
  padrão (amostral, ddof=1) dos 10 valores.
* A padronização (StandardScaler) fica DENTRO de um Pipeline, logo é ajustada
  apenas nos 9 folds de treino de cada rodada (sem vazamento de dados).
* Análises extras para a discussão:
    - teste de Friedman + Wilcoxon pareado entre os melhores modelos;
    - matrizes de confusão e curvas ROC (predições fora-do-fold);
    - ablação do pré-processamento (sem padronização / sem transformação log);
    - importância dos atributos (Árvore de Decisão).

Entrada : data/processed/netflix_churn_encoded.csv (gerado por eda_preprocessamento.py)
Saídas  : resultados/*.csv, resultados/relatorio_experimentos.txt, figuras/09..15*.png

Uso: python experimentos.py
"""

import time
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera figuras sem abrir janela
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_validate)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ----------------------------------------------------------------------------
# Configurações gerais
# ----------------------------------------------------------------------------
SEMENTE = 42
N_FOLDS = 10
BASE = Path(__file__).resolve().parent
ARQ_DADOS = BASE / "data" / "processed" / "netflix_churn_encoded.csv"
DIR_RES = BASE / "resultados"
DIR_FIG = BASE / "figuras"
DIR_RES.mkdir(exist_ok=True)
DIR_FIG.mkdir(exist_ok=True)

# Métricas coletadas em cada fold (nomes do scikit-learn -> nome no relatório)
METRICAS = {
    "accuracy": "Acurácia",
    "precision": "Precisão",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "AUC",
}
METRICA_SELECAO = "f1"  # critério para escolher a melhor combinação de cada algoritmo

# Atributos numéricos (padronizados). Os demais são dummies 0/1 do one-hot e
# ficam como estão, mantendo a mesma decisão da Entrega 1.
NUMERICOS = ["age", "subscription_type", "watch_hours", "last_login_days",
             "number_of_profiles", "avg_watch_time_per_day"]
COLS_LOG = ["watch_hours", "avg_watch_time_per_day"]  # receberam log1p na Entrega 1

saida_txt = []


def log(texto=""):
    """Imprime no terminal e guarda no relatório texto."""
    print(texto)
    saida_txt.append(str(texto))


def titulo(texto):
    log("\n" + "=" * 78 + f"\n{texto}\n" + "=" * 78)


# ----------------------------------------------------------------------------
# 1. Dados
# ----------------------------------------------------------------------------
dados = pd.read_csv(ARQ_DADOS)
X = dados.drop(columns="churned")
y = dados["churned"].to_numpy()

titulo("1. DADOS")
log(f"Arquivo: {ARQ_DADOS.relative_to(BASE)}")
log(f"Exemplos: {X.shape[0]} | Atributos: {X.shape[1]} | Classes: {len(np.unique(y))}")
contagem = pd.Series(y).value_counts().sort_index()
for c, n in contagem.items():
    log(f"  Classe {c}: {n} ({n / len(y):.1%})")
log(f"Classe majoritária: {contagem.idxmax()} com {contagem.max()} exemplos")

# Partições estratificadas fixas: TODOS os algoritmos usam exatamente os mesmos folds
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMENTE)


def montar_pipeline(classificador, padronizar=True):
    """Pré-processador (padroniza só os numéricos) + classificador."""
    escala = StandardScaler() if padronizar else "passthrough"
    prep = ColumnTransformer([("num", escala, NUMERICOS)], remainder="passthrough")
    return Pipeline([("prep", prep), ("clf", classificador)])


# ----------------------------------------------------------------------------
# 2. Algoritmos e grades de hiperparâmetros
# ----------------------------------------------------------------------------
# Compatibilidade: a partir do scikit-learn 1.8 o parâmetro "penalty" foi
# depreciado e basta informar l1_ratio; em versões anteriores é preciso
# declarar penalty="elasticnet" para que l1_ratio seja usado.
_versao = tuple(int(v) for v in sklearn.__version__.split(".")[:2])
LR_EXTRA = {} if _versao >= (1, 8) else {"penalty": "elasticnet"}

# Cada grade gera >= 3 combinações (exigência do projeto). Os prefixos "clf__"
# indicam que o parâmetro pertence ao classificador dentro do Pipeline.
ALGORITMOS = {
    "Árvore de Decisão": (
        DecisionTreeClassifier(random_state=SEMENTE),
        {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [3, 5, 8, None],
            "clf__min_samples_leaf": [1, 10],
        },
    ),
    "KNN": (
        KNeighborsClassifier(),
        {
            "clf__n_neighbors": [3, 5, 11, 21, 51, 101],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],  # 1 = Manhattan, 2 = Euclidiana
        },
    ),
    "Naive Bayes": (
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]},
    ),
    "Regressão Logística": (
        # l1_ratio: 0 = regularização L2 (ridge), 1 = L1 (lasso), 0.5 = elastic-net
        LogisticRegression(solver="saga", max_iter=5000, random_state=SEMENTE,
                           **LR_EXTRA),
        {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__l1_ratio": [0.0, 0.5, 1.0],
        },
    ),
    "SVM": (
        SVC(random_state=SEMENTE),
        [
            {"clf__kernel": ["linear"], "clf__C": [0.1, 1, 10]},
            {"clf__kernel": ["rbf"], "clf__C": [0.1, 1, 10],
             "clf__gamma": ["scale", 0.01, 0.1]},
            {"clf__kernel": ["poly"], "clf__C": [1], "clf__degree": [2, 3]},
        ],
    ),
    "MLP": (
        MLPClassifier(max_iter=1000, early_stopping=True, n_iter_no_change=20,
                      random_state=SEMENTE),
        {
            "clf__hidden_layer_sizes": [(16,), (64,), (64, 32)],
            "clf__activation": ["relu", "tanh"],
            "clf__alpha": [1e-4, 1e-2],
        },
    ),
}


def formatar_params(params):
    """{'clf__C': 1, ...} -> 'C=1, ...' (texto legível para tabelas)."""
    return ", ".join(f"{k.replace('clf__', '')}={v}" for k, v in params.items())


# ----------------------------------------------------------------------------
# 3. Grid search com 10-fold estratificado
# ----------------------------------------------------------------------------
titulo("2. GRID SEARCH + 10-FOLD CROSS-VALIDATION ESTRATIFICADO")

linhas_combinacoes = []  # uma linha por (algoritmo, combinação): média e dp
linhas_folds = []        # uma linha por (algoritmo, combinação, fold)
melhores = {}            # algoritmo -> melhores parâmetros

for nome, (clf, grade) in ALGORITMOS.items():
    inicio = time.time()
    busca = GridSearchCV(
        montar_pipeline(clf), grade, cv=skf, scoring=list(METRICAS),
        refit=METRICA_SELECAO, n_jobs=-1,
    )
    busca.fit(X, y)
    res = busca.cv_results_
    n_comb = len(res["params"])

    for i, params in enumerate(res["params"]):
        linha = {"algoritmo": nome, "parametros": formatar_params(params)}
        for m in METRICAS:
            # valores da métrica m nos 10 folds para a combinação i
            valores = np.array([res[f"split{k}_test_{m}"][i] for k in range(N_FOLDS)])
            linha[f"{m}_media"] = valores.mean()
            linha[f"{m}_dp"] = valores.std(ddof=1)
            for k, v in enumerate(valores):
                linhas_folds.append({"algoritmo": nome, "parametros": linha["parametros"],
                                     "fold": k + 1, "metrica": m, "valor": v})
        linha["tempo_ajuste_s"] = res["mean_fit_time"][i]
        linhas_combinacoes.append(linha)

    melhores[nome] = busca.best_params_
    log(f"\n{nome}: {n_comb} combinações x {N_FOLDS} folds "
        f"({time.time() - inicio:.1f}s)")
    log(f"  Melhor ({METRICA_SELECAO}): {formatar_params(busca.best_params_)} "
        f"-> F1 = {busca.best_score_:.4f}")

tab_comb = pd.DataFrame(linhas_combinacoes)
tab_folds = pd.DataFrame(linhas_folds)
tab_comb.to_csv(DIR_RES / "todas_combinacoes.csv", index=False, float_format="%.4f")
tab_folds.to_csv(DIR_RES / "metricas_por_fold.csv", index=False, float_format="%.4f")

# Tabela completa, legível: "média ± dp" para cada combinação
titulo("3. DESEMPENHO DE CADA COMBINAÇÃO (média ± desvio padrão nos 10 folds)")
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.max_rows", 200)
for nome in ALGORITMOS:
    sub = tab_comb[tab_comb["algoritmo"] == nome].copy()
    exibir = pd.DataFrame({"parâmetros": sub["parametros"]})
    for m, rotulo in METRICAS.items():
        exibir[rotulo] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(sub[f"{m}_media"], sub[f"{m}_dp"])]
    log(f"\n--- {nome} ---")
    log(exibir.to_string(index=False))
    # média geral do algoritmo considerando todas as suas combinações
    medias = "  ".join(f"{METRICAS[m]}={sub[f'{m}_media'].mean():.4f}" for m in METRICAS)
    log(f"Média geral das combinações: {medias}")

# ----------------------------------------------------------------------------
# 4. Melhor configuração de cada algoritmo
# ----------------------------------------------------------------------------
titulo("4. MELHOR CONFIGURAÇÃO DE CADA ALGORITMO")
linhas_melhores = []
for nome, params in melhores.items():
    linha = tab_comb[(tab_comb["algoritmo"] == nome) &
                     (tab_comb["parametros"] == formatar_params(params))].iloc[0]
    linhas_melhores.append(linha)
tab_melhores = pd.DataFrame(linhas_melhores).reset_index(drop=True)

# Linha "Média geral" exigida pelo enunciado (média entre os algoritmos)
media_geral = {"algoritmo": "Média geral", "parametros": "-"}
for m in METRICAS:
    media_geral[f"{m}_media"] = tab_melhores[f"{m}_media"].mean()
    media_geral[f"{m}_dp"] = tab_melhores[f"{m}_media"].std(ddof=1)
tab_melhores_csv = pd.concat([tab_melhores, pd.DataFrame([media_geral])], ignore_index=True)
tab_melhores_csv.to_csv(DIR_RES / "melhores_por_algoritmo.csv", index=False, float_format="%.4f")

exibir = pd.DataFrame({"algoritmo": tab_melhores_csv["algoritmo"],
                       "parâmetros": tab_melhores_csv["parametros"]})
for m, rotulo in METRICAS.items():
    exibir[rotulo] = [f"{a:.4f} ± {b:.4f}" for a, b in
                      zip(tab_melhores_csv[f"{m}_media"], tab_melhores_csv[f"{m}_dp"])]
log(exibir.to_string(index=False))

# Modelos finais (pipelines com os melhores parâmetros) usados nas análises seguintes
modelos_finais = {
    nome: montar_pipeline(clone(ALGORITMOS[nome][0])).set_params(**melhores[nome])
    for nome in ALGORITMOS
}

# ----------------------------------------------------------------------------
# 5. Testes estatísticos (Friedman + Wilcoxon pareado) sobre F1 por fold
# ----------------------------------------------------------------------------
titulo("5. TESTES ESTATÍSTICOS (F1 dos melhores modelos nos 10 folds)")
f1_folds = {}
for nome, params in melhores.items():
    sel = tab_folds[(tab_folds["algoritmo"] == nome) &
                    (tab_folds["parametros"] == formatar_params(params)) &
                    (tab_folds["metrica"] == "f1")].sort_values("fold")
    f1_folds[nome] = sel["valor"].to_numpy()
matriz_f1 = pd.DataFrame(f1_folds)

estat, p_fried = stats.friedmanchisquare(*[matriz_f1[c] for c in matriz_f1])
log(f"Friedman: chi2 = {estat:.3f}, p-valor = {p_fried:.2e}")
ranks_medios = matriz_f1.rank(axis=1, ascending=False).mean().sort_values()
log("Ranking médio (1 = melhor):")
log(ranks_medios.round(2).to_string())

log("\nWilcoxon pareado (p-valores; correção de Holm para comparações múltiplas):")
pares = list(combinations(matriz_f1.columns, 2))
p_brutos = [stats.wilcoxon(matriz_f1[a], matriz_f1[b]).pvalue for a, b in pares]
# correção de Holm-Bonferroni
ordem = np.argsort(p_brutos)
p_holm = np.empty(len(p_brutos))
acumulado = 0.0
for pos, idx in enumerate(ordem):
    acumulado = max(acumulado, min(1.0, (len(p_brutos) - pos) * p_brutos[idx]))
    p_holm[idx] = acumulado
tab_wilcoxon = pd.DataFrame({"modelo_A": [a for a, _ in pares],
                             "modelo_B": [b for _, b in pares],
                             "p_valor": p_brutos, "p_holm": p_holm})
tab_wilcoxon["significativo_5%"] = tab_wilcoxon["p_holm"] < 0.05
tab_wilcoxon.to_csv(DIR_RES / "testes_wilcoxon.csv", index=False, float_format="%.4g")
log(tab_wilcoxon.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ----------------------------------------------------------------------------
# 6. Ablação do pré-processamento (impacto de cada etapa)
# ----------------------------------------------------------------------------
# Com os melhores parâmetros de cada algoritmo e os MESMOS folds, comparamos:
#   (a) pipeline completo (log + padronização);
#   (b) sem padronização;
#   (c) sem transformação log (desfeita com expm1, inverso exato de log1p).
titulo("6. ABLAÇÃO DO PRÉ-PROCESSAMENTO (F1 e acurácia, média ± dp)")
X_sem_log = X.copy()
X_sem_log[COLS_LOG] = np.expm1(X_sem_log[COLS_LOG])

cenarios = {
    "Completo": (X, True),
    "Sem padronização": (X, False),
    "Sem log": (X_sem_log, True),
}
linhas_abl = []
for nome in ALGORITMOS:
    for cenario, (Xc, padronizar) in cenarios.items():
        modelo = montar_pipeline(clone(ALGORITMOS[nome][0]), padronizar)
        modelo.set_params(**melhores[nome])
        cv = cross_validate(modelo, Xc, y, cv=skf, scoring=["accuracy", "f1"], n_jobs=-1)
        linhas_abl.append({
            "algoritmo": nome, "cenario": cenario,
            "accuracy_media": cv["test_accuracy"].mean(),
            "accuracy_dp": cv["test_accuracy"].std(ddof=1),
            "f1_media": cv["test_f1"].mean(), "f1_dp": cv["test_f1"].std(ddof=1),
        })
tab_abl = pd.DataFrame(linhas_abl)
tab_abl.to_csv(DIR_RES / "ablacao_preprocessamento.csv", index=False, float_format="%.4f")
pivot_abl = tab_abl.pivot(index="algoritmo", columns="cenario", values="f1_media")[list(cenarios)]
log("F1 médio:")
log(pivot_abl.round(4).to_string())

# ----------------------------------------------------------------------------
# 7. Figuras
# ----------------------------------------------------------------------------
titulo("7. FIGURAS")
sns.set_theme(style="whitegrid")
ordem_alg = list(ALGORITMOS)
cores = dict(zip(ordem_alg, sns.color_palette("tab10", len(ordem_alg))))

# 09 - barras com média ± dp das métricas dos melhores modelos
fig, ax = plt.subplots(figsize=(11, 5))
largura = 0.8 / len(ordem_alg)
x_pos = np.arange(len(METRICAS))
for j, nome in enumerate(ordem_alg):
    linha = tab_melhores[tab_melhores["algoritmo"] == nome].iloc[0]
    ax.bar(x_pos + j * largura, [linha[f"{m}_media"] for m in METRICAS], largura,
           yerr=[linha[f"{m}_dp"] for m in METRICAS], capsize=2, label=nome, color=cores[nome])
ax.set_xticks(x_pos + largura * (len(ordem_alg) - 1) / 2)
ax.set_xticklabels(METRICAS.values())
ax.set_ylim(0.75, 1.02)
ax.set_ylabel("Média nos 10 folds")
ax.set_title("Melhor configuração de cada algoritmo (barras = desvio padrão)")
ax.legend(ncol=6, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
fig.tight_layout()
fig.savefig(DIR_FIG / "09_comparacao_melhores.png", dpi=150)
plt.close(fig)

# 10 - boxplot do F1 por fold (variabilidade entre folds)
fig, ax = plt.subplots(figsize=(10, 4.5))
sns.boxplot(data=matriz_f1[ordem_alg], ax=ax, palette=[cores[a] for a in ordem_alg])
sns.stripplot(data=matriz_f1[ordem_alg], ax=ax, color="black", size=3, alpha=0.6)
ax.set_ylabel("F1 no fold")
ax.set_title("Distribuição do F1 nos 10 folds (melhor configuração)")
fig.tight_layout()
fig.savefig(DIR_FIG / "10_boxplot_f1_folds.png", dpi=150)
plt.close(fig)

# 11 - sensibilidade aos hiperparâmetros: F1 de todas as combinações
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, nome in zip(axes.ravel(), ordem_alg):
    sub = tab_comb[tab_comb["algoritmo"] == nome].reset_index(drop=True)
    destaque = sub["parametros"] == formatar_params(melhores[nome])
    ax.errorbar(sub.index + 1, sub["f1_media"], yerr=sub["f1_dp"], fmt="o",
                color=cores[nome], ecolor="gray", capsize=2, markersize=4)
    ax.scatter(sub.index[destaque] + 1, sub.loc[destaque, "f1_media"], s=120,
               facecolors="none", edgecolors="red", linewidths=1.5, label="melhor")
    ax.set_title(f"{nome} ({len(sub)} combinações)")
    # numeração = ordem das linhas de resultados/todas_combinacoes.csv
    # (Tabela 6, apêndice do artigo)
    ax.set_xlabel("Nº da combinação (ordem da Tabela 6)")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylabel("F1 (média ± dp)")
    ax.legend(fontsize=8, loc="lower right")
fig.suptitle("Sensibilidade de cada algoritmo aos hiperparâmetros")
fig.tight_layout()
fig.savefig(DIR_FIG / "11_sensibilidade_parametros.png", dpi=150)
plt.close(fig)

# 12 e 13 - matrizes de confusão e curvas ROC com predições fora-do-fold
fig_cm, axes_cm = plt.subplots(2, 3, figsize=(13, 8))
fig_roc, ax_roc = plt.subplots(figsize=(6.5, 6))
linhas_cm = []
for ax, nome in zip(axes_cm.ravel(), ordem_alg):
    modelo = modelos_finais[nome]
    pred = cross_val_predict(modelo, X, y, cv=skf, n_jobs=-1)
    metodo = "predict_proba" if hasattr(modelo.named_steps["clf"], "predict_proba") \
        and not isinstance(modelo.named_steps["clf"], SVC) else "decision_function"
    score = cross_val_predict(modelo, X, y, cv=skf, method=metodo, n_jobs=-1)
    if score.ndim == 2:
        score = score[:, 1]
    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()
    linhas_cm.append({"algoritmo": nome, "VN": tn, "FP": fp, "FN": fn, "VP": tp})
    ConfusionMatrixDisplay(cm, display_labels=["Ficou (0)", "Cancelou (1)"]).plot(
        ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.set_title(nome)
    fpr, tpr, _ = roc_curve(y, score)
    ax_roc.plot(fpr, tpr, color=cores[nome], label=f"{nome} (AUC={roc_auc_score(y, score):.3f})")
fig_cm.suptitle("Matrizes de confusão (predições fora-do-fold, 10-fold estratificado)")
fig_cm.tight_layout()
fig_cm.savefig(DIR_FIG / "12_matrizes_confusao.png", dpi=150)
plt.close(fig_cm)
ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
ax_roc.set_xlabel("Taxa de falsos positivos")
ax_roc.set_ylabel("Taxa de verdadeiros positivos")
ax_roc.set_title("Curvas ROC (predições fora-do-fold)")
ax_roc.legend(fontsize=9, loc="lower right")
fig_roc.tight_layout()
fig_roc.savefig(DIR_FIG / "13_curvas_roc.png", dpi=150)
plt.close(fig_roc)
pd.DataFrame(linhas_cm).to_csv(DIR_RES / "matrizes_confusao.csv", index=False)

# 14 - ablação do pré-processamento
fig, ax = plt.subplots(figsize=(10, 4.5))
plot_abl = tab_abl.copy()
sns.barplot(data=plot_abl, x="algoritmo", y="f1_media", hue="cenario", ax=ax,
            order=ordem_alg, hue_order=list(cenarios))
ax.set_ylim(0.7, 1.0)
ax.set_xlabel("")
ax.set_ylabel("F1 médio (10 folds)")
ax.set_title("Impacto do pré-processamento (melhores parâmetros de cada algoritmo)")
ax.legend(title="", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
fig.tight_layout()
fig.savefig(DIR_FIG / "14_ablacao_preprocessamento.png", dpi=150)
plt.close(fig)

# 15 - importância dos atributos (Árvore de Decisão treinada na base toda)
arvore = clone(modelos_finais["Árvore de Decisão"]).fit(X, y)
nomes_saida = arvore.named_steps["prep"].get_feature_names_out()
nomes_saida = [n.split("__", 1)[1] for n in nomes_saida]
imp = pd.Series(arvore.named_steps["clf"].feature_importances_, index=nomes_saida)
imp = imp.sort_values(ascending=False).head(10)
imp.to_csv(DIR_RES / "importancia_atributos_arvore.csv", header=["importancia"])
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(x=imp.values, y=imp.index, ax=ax, color=cores["Árvore de Decisão"])
ax.set_xlabel("Importância (redução de impureza)")
ax.set_title("10 atributos mais importantes - Árvore de Decisão")
fig.tight_layout()
fig.savefig(DIR_FIG / "15_importancia_atributos_arvore.png", dpi=150)
plt.close(fig)
log("Importância dos atributos (Árvore):")
log(imp.round(4).to_string())
log(f"Profundidade da árvore final: {arvore.named_steps['clf'].get_depth()} | "
    f"folhas: {arvore.named_steps['clf'].get_n_leaves()}")

# Regras dos 3 primeiros níveis, com limiares na escala ORIGINAL dos atributos
# (árvore ajustada sem padronização; a árvore é invariante à escala).
arvore_txt = montar_pipeline(clone(ALGORITMOS["Árvore de Decisão"][0]), padronizar=False)
arvore_txt.set_params(**melhores["Árvore de Decisão"]).fit(X_sem_log, y)
log("\nRegras da árvore (até 3 níveis; watch_hours e avg_watch_time_per_day sem log):")
# o ColumnTransformer reordena as colunas (numéricos primeiro): usar os nomes de saída
nomes_txt = [n.split("__", 1)[1] for n in arvore_txt.named_steps["prep"].get_feature_names_out()]
log(export_text(arvore_txt.named_steps["clf"], feature_names=nomes_txt, max_depth=3))

log("\nFiguras salvas: figuras/09 a 15. Tabelas salvas em resultados/.")
(DIR_RES / "relatorio_experimentos.txt").write_text("\n".join(saida_txt), encoding="utf-8")
