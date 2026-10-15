# Projeto 2ª VA – Inteligência Artificial (UFRPE/UABJ) – Entrega 1

**Disciplina:** Inteligência Artificial — Prof. Anderson Cavalcanti
**Grupo:**
- Adriano da Silva Júnior
- Daniel Ronald Gomes da Silva
- Fernanda Felix Fernandes
- Lucas Gonçalo dos Santos
- Rayslla Nayara Nascimento Leite

**Entrega:** 13/10

Objetivo do projeto: comparar algoritmos de classificação. Esta 1ª entrega cobre a
**definição, análise exploratória e pré-processamento** do dataset escolhido.

## 1. Dataset escolhido

| Item | Valor |
|---|---|
| Nome | Netflix Customer Churn |
| Fonte | Kaggle (https://www.kaggle.com/datasets) |
| Tarefa | Classificação binária — prever se o cliente cancelou (`churned`) |
| Exemplos | 5.000 (> 100 exigidos) |
| Colunas | 14 = 1 identificador + 12 atributos preditores + 1 alvo |

Atributos: `age`, `gender`, `subscription_type`, `watch_hours`, `last_login_days`,
`region`, `device`, `monthly_fee`, `payment_method`, `number_of_profiles`,
`avg_watch_time_per_day`, `favorite_genre`.

## 2. Estrutura do repositório

```
├── data/
│   ├── original/netflix_customer_churn.csv     # base original
│   └── processed/
│       ├── netflix_churn_encoded.csv           # codificada (sem escala)
│       └── netflix_churn_scaled.csv            # codificada + padronizada
├── figuras/                                    # gráficos gerados pela EDA
├── eda_preprocessamento.py                     # código-fonte comentado
├── relatorio_eda.txt                           # saída completa da análise
└── requirements.txt
```

Para reproduzir:

```bash
pip install -r requirements.txt
python eda_preprocessamento.py
```

## 3. Principais resultados da análise

**Tipos e qualidade dos dados**
- 6 atributos numéricos (`age`, `watch_hours`, `last_login_days`, `monthly_fee`,
  `number_of_profiles`, `avg_watch_time_per_day`) e 6 categóricos.
- **Nenhum valor ausente** e **nenhuma linha duplicada**.

**Balanceamento das classes**
- Classe 0 (permaneceu): 2.485 (49,7%) · Classe 1 (cancelou): 2.515 (50,3%).
- Base **balanceada** → não é necessário SMOTE/undersampling.

**Distribuição dos atributos**
- `age`, `last_login_days` e `number_of_profiles` são aproximadamente uniformes.
- `watch_hours` (assimetria 2,26) e `avg_watch_time_per_day` (assimetria 15,8) são
  fortemente assimétricos à direita, com outliers pelo critério IQR (4,8% e 11,0%).
- Categóricos têm categorias com frequências bem equilibradas.

**Relação com o alvo**
- Mais relevantes: `watch_hours` (r = −0,48), `last_login_days` (r = +0,47),
  `avg_watch_time_per_day` (r = −0,27), `number_of_profiles` (r = −0,16).
- Plano Basic tem 61,8% de churn vs. ~44–45% nos demais.
- Pagamento com Crypto/Gift Card: ~58–60% de churn vs. ~44% com cartão.
- `age`, `gender`, `region`, `device` e `favorite_genre` têm pouca relação com o churn.

**Redundâncias detectadas**
- `monthly_fee` é determinado 1:1 por `subscription_type` (8,99 / 13,99 / 17,99).
- `avg_watch_time_per_day` ≈ `watch_hours / (last_login_days + 1)` (erro médio 0,002),
  ou seja, é um atributo derivado.

## 4. Pré-processamentos aplicados

| # | Pré-processamento | Justificativa |
|---|---|---|
| 1 | Remoção de `customer_id` | Identificador único, sem valor preditivo |
| 2 | Remoção de `monthly_fee` | Redundante com `subscription_type` |
| 3 | `log(1+x)` em `watch_hours` e `avg_watch_time_per_day` | Reduz assimetria e o efeito dos outliers sem descartar exemplos (valores plausíveis) |
| 4 | Codificação ordinal de `subscription_type` (Basic=0, Standard=1, Premium=2) | Existe ordem natural entre os planos |
| 5 | One-hot em `gender`, `region`, `device`, `payment_method`, `favorite_genre` | Categóricos nominais, sem ordem |
| 6 | Padronização (StandardScaler) dos atributos numéricos | Necessário para KNN, SVM, MLP, Regressão Logística |

Não foram necessários: imputação de ausentes, remoção de duplicatas, reamostragem.

Resultado: **5.000 exemplos × 32 atributos + 1 alvo**.

> Observação para a Entrega 2: o `StandardScaler` deve ser ajustado somente no conjunto
> de treino (via `Pipeline` do scikit-learn) para evitar vazamento de dados. A versão
> `netflix_churn_scaled.csv` foi gerada sobre a base inteira apenas para fins de análise.

## 5. Figuras

| Arquivo | Conteúdo |
|---|---|
| `01_histogramas_numericos.png` | Distribuição dos numéricos |
| `02_boxplots_numericos.png` | Outliers |
| `03_barras_categoricos.png` | Frequência das categorias |
| `04_balanceamento_classes.png` | Balanceamento do alvo |
| `05_matriz_correlacao.png` | Correlação entre numéricos e alvo |
| `06_numericos_por_classe.png` | Numéricos separados por classe |
| `07_churn_por_categoria.png` | Taxa de churn por categoria |
| `08_apos_transformacao_log.png` | Distribuição após `log(1+x)` |
