# House Prices — Pipeline de ML em Produção

Pipeline de MLOps aplicado ao modelo LightGBM do [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques), cobrindo treinamento, serving via API REST e monitoramento de data drift.

---

## Resultado

| Métrica | Valor |
|---|---|
| RMSLE CV (5-fold) | **0.15332 ± 0.01896** |
| RMSLE Kaggle | **0.12436** |
| Algoritmo | LightGBM |
| Modelo registrado | HousePricesLightGBM v1 (MLflow) |
| Amostras de treino | 1.460 |
| Features | 10 |

---

## Estrutura

```
house-prices-mlops/
├── data/
│   ├── train.csv          ← copiar do Kaggle
│   └── test.csv           ← copiar do Kaggle
├── models/                ← gerado pelo train.py
│   ├── house_prices_model.pkl
│   └── features.pkl
├── monitoring/
│   ├── drift.py           ← Evidently
│   └── reference.csv      ← gerado pelo train.py
├── train.py               ← treina, loga no MLflow e serializa o modelo
├── main.py                ← FastAPI
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Como Executar

### Pré-requisitos

- Docker Desktop instalado e rodando
- Arquivos `train.csv` e `test.csv` na pasta `data/` (baixar do [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data))

### 1. Treinar o modelo

```bash
pip install -r requirements.txt
python train.py
```

O script treina o LightGBM, loga parâmetros e métricas no MLflow e salva o modelo em `models/house_prices_model.pkl`.

### 2. Subir a API

```bash
docker-compose up --build
```

| Serviço | URL |
|---|---|
| FastAPI (Swagger) | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |
| MLflow UI | http://localhost:5000 |

### 3. Fazer uma predição

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "OverallQual": 7,
    "GrLivArea": 1500,
    "GarageCars": 2,
    "TotalBsmtSF": 800,
    "YearBuilt": 2005,
    "YearRemodAdd": 2010,
    "LotArea": 8500,
    "Fireplaces": 1,
    "TotalBaths": 2.5,
    "TotalSF": 2300
  }'
```

Resposta:

```json
{
  "predicted_price": 196485.11
}
```

### 4. Gerar relatório de data drift

```bash
python monitoring/drift.py
```

O relatório HTML interativo é salvo em `monitoring/drift_report.html`. Abra no navegador para visualizar a análise feature por feature.

---

## Etapas do Projeto

### 1. Treinamento — `train.py`

O modelo é treinado com 10 features selecionadas pela importância identificada no Projeto 1 (Optuna + NSGA-II).

| Feature | Descrição | Importância |
|---|---|---|
| `LotArea` | Área do lote (sqft) | 1622 |
| `TotalSF` | Área total: subsolo + área habitável | 1403 |
| `TotalBsmtSF` | Área total do subsolo (sqft) | 1255 |
| `GrLivArea` | Área habitável acima do solo (sqft) | 1142 |
| `YearBuilt` | Ano de construção | 910 |
| `YearRemodAdd` | Ano da última reforma | 767 |
| `TotalBaths` | Total de banheiros (lavabo = 0.5) | 409 |
| `OverallQual` | Qualidade geral da construção (1-10) | 396 |
| `GarageCars` | Capacidade da garagem em carros | 206 |
| `Fireplaces` | Número de lareiras | 185 |

O target é transformado com `log1p(SalePrice)` antes do treino — minimizar RMSE no espaço log equivale a minimizar RMSLE, reduzindo o efeito de casas de alto valor na função de perda.

### 2. Versionamento — MLflow

Todos os parâmetros, métricas e feature importance são logados automaticamente no MLflow a cada execução do `train.py`. O Model Registry mantém o histórico de versões com rastreabilidade completa — qual run gerou cada versão, quais métricas tinha, quando foi criado.

```
Experimento 'house-prices'
    └── Run: lightgbm_optuna_params
          ├── Parâmetros: n_estimators=500, learning_rate=0.05, max_depth=8
          ├── Métricas:   rmsle_cv=0.15332, rmsle_kaggle=0.12436
          ├── Feature importance: LotArea=1622, TotalSF=1403, ...
          └── → Model Registry: HousePricesLightGBM v1
```

### 3. Serving — FastAPI

A API carrega o modelo serializado na inicialização e expõe três endpoints:

| Endpoint | Método | Descrição |
|---|---|---|
| `/predict` | POST | Predição de preço dado as features do imóvel |
| `/health` | GET | Verifica se o modelo está carregado |
| `/docs` | GET | Documentação interativa (Swagger UI) |

### 4. Monitoramento — Evidently

O script `drift.py` compara a distribuição dos dados de produção com os dados de treino (referência) e detecta mudanças estatísticas que podem indicar degradação do modelo.

| Informação no relatório | Descrição |
|---|---|
| Status por feature | Drift detectado ou não (teste KS) |
| Distribuições sobrepostas | Referência vs. produção |
| Percentual com drift | Share de features afetadas |

Quando drift é detectado, o modelo deve ser retreinado com dados mais recentes.

---

## Tecnologias

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Evidently](https://img.shields.io/badge/Evidently-FF6B35?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

---

*Juliana Burato — 2026*
