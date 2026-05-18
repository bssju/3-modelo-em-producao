import pandas as pd
import numpy as np
import pickle
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("🚀 Iniciando treinamento do modelo...")

# ── Configuração do MLflow ────────────────────────────────────────────────────
# O tracking server roda no container mlflow (docker-compose)
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MODEL_NAME          = 'HousePricesLightGBM'

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment('house-prices')

os.makedirs('models', exist_ok=True)
os.makedirs('monitoring', exist_ok=True)

# ── Dados ─────────────────────────────────────────────────────────────────────
print("📊 Carregando dados...")
df = pd.read_csv('data/train.csv')

features = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
    'YearBuilt', 'YearRemodAdd', 'LotArea', 'Fireplaces',
    'TotalBaths', 'TotalSF'
]

# Feature engineering — mesmas features usadas no notebook Kaggle
df['TotalBaths'] = (df['FullBath'] + 0.5 * df['HalfBath'] +
                    df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea']

# Tratamento de NaN
for col in features:
    if col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(
                df[col].mode()[0] if len(df[col].mode()) > 0 else 0
            )

X = df[features].copy()
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

# log1p no target: minimizar RMSE no espaço log equivale a minimizar RMSLE
y = np.log1p(df['SalePrice'])

print(f"📈 Features: {features}")
print(f"📏 Dados: {X.shape[0]:,} amostras, {X.shape[1]} features")

# ── Hiperparâmetros ───────────────────────────────────────────────────────────
# Parâmetros otimizados pelo Optuna no Projeto 1 (RMSLE Kaggle: 0.12436)
params = {
    'n_estimators':  500,
    'learning_rate': 0.05,
    'max_depth':     8,
    'random_state':  42,
    'verbose':       -1,
}
CV_FOLDS = 5

# ── Cross-validation ──────────────────────────────────────────────────────────
# CV antes do treino final para estimar o erro de generalização
print(f"📊 Avaliando com CV {CV_FOLDS}-fold...")
kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
fold_scores = []
for tr_idx, val_idx in kf.split(X):
    m = lgb.LGBMRegressor(**params)
    m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    fold_scores.append(
        np.sqrt(mean_squared_error(y.iloc[val_idx], m.predict(X.iloc[val_idx])))
    )
rmsle_cv  = float(np.mean(fold_scores))
rmsle_std = float(np.std(fold_scores))
print(f"  RMSLE CV: {rmsle_cv:.5f} ± {rmsle_std:.5f}")

# ── Treino final ──────────────────────────────────────────────────────────────
print("🏋️  Treinando modelo final...")
model = lgb.LGBMRegressor(**params)
model.fit(X, y)

y_pred      = model.predict(X)
rmsle_train = float(np.sqrt(mean_squared_error(y, y_pred)))
print(f"  RMSLE treino: {rmsle_train:.5f}")

# ── MLflow — logar experimento ────────────────────────────────────────────────
print("📝 Logando no MLflow...")
with mlflow.start_run(run_name='lightgbm_optuna_params') as run:
    # Parâmetros
    mlflow.log_params(params)
    mlflow.log_param('cv_folds',   CV_FOLDS)
    mlflow.log_param('n_features', len(features))
    mlflow.log_param('n_train',    X.shape[0])

    # Métricas
    mlflow.log_metric('rmsle_cv',      rmsle_cv)
    mlflow.log_metric('rmsle_cv_std',  rmsle_std)
    mlflow.log_metric('rmsle_train',   rmsle_train)
    # Score público confirmado no Kaggle (Projeto 1)
    mlflow.log_metric('rmsle_kaggle',  0.12436)

    # Feature importance — útil para auditar o modelo no MLflow UI
    for fname, imp in zip(features, model.feature_importances_):
        mlflow.log_metric(f'imp_{fname}', int(imp))

    # Registrar modelo no MLflow para versionamento
    mlflow.sklearn.log_model(model, name='model')
    run_id = run.info.run_id
    print(f"  run_id: {run_id}")

# Registrar no Model Registry
model_uri = f'runs:/{run_id}/model'
mv = mlflow.register_model(model_uri, MODEL_NAME)
print(f"  Modelo registrado: {MODEL_NAME} v{mv.version}")

# ── Pickle — para a API FastAPI ───────────────────────────────────────────────
# A API carrega o modelo via pickle (mais simples para serving local)
# O MLflow serve para rastreabilidade e versionamento do experimento
model_path = 'models/house_prices_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

features_path = 'models/features.pkl'
with open(features_path, 'wb') as f:
    pickle.dump(features, f)

# ── Dados de referência para monitoramento de drift ───────────────────────────
ref_df = X.copy()
ref_df['target'] = y.values
ref_df.to_csv('monitoring/reference.csv', index=False)

# ── Resumo ────────────────────────────────────────────────────────────────────
print()
print('=' * 50)
print(f'  RMSLE CV     : {rmsle_cv:.5f} ± {rmsle_std:.5f}')
print(f'  RMSLE Kaggle : 0.12436')
print(f'  MLflow UI    : {MLFLOW_TRACKING_URI}')
print(f'  Modelo       : {MODEL_NAME} v{mv.version}')
print(f'  Pickle       : {model_path}')
print('=' * 50)
print("🎉 Treinamento concluído com sucesso!")
