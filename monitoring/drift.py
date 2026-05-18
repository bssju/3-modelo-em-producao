"""
monitoring/drift.py — Monitoramento de data drift com Evidently

Compara a distribuição dos dados de produção com os dados de referência
(treino) e gera um relatório HTML interativo.

Execução:
    python monitoring/drift.py

O script simula dados de produção com drift para demonstração.
Em produção real, os dados viriam de logs da API ou de um banco de dados.

Saída:
    monitoring/drift_report.html — relatório interativo do Evidently
"""

import os
import warnings
import numpy as np
import pandas as pd
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset

warnings.filterwarnings('ignore')

# ── Configuração ──────────────────────────────────────────────────────────────
MONITORING_DIR  = os.path.dirname(os.path.abspath(__file__))
REFERENCE_PATH  = os.path.join(MONITORING_DIR, 'reference.csv')
REPORT_PATH     = os.path.join(MONITORING_DIR, 'drift_report.html')

# Features preferidas para monitorar (quando disponíveis no reference.csv)
# Geradas pelo train.py com dados reais do Kaggle
TOP_FEATURES = [
    'OverallQual', 'GrLivArea', 'TotalSF', 'GarageCars', 'TotalBath',
    'TotalBsmtSF', 'HouseAge', 'YearsSinceRemod', 'Fireplaces', 'LotArea',
    'HasGarage', 'HasBasement', 'HasFireplace', 'WasRemodeled',
]


def select_monitor_columns(reference: pd.DataFrame,
                            top_features: list,
                            max_cols: int = 15) -> list:
    """Seleciona as colunas a monitorar no relatório.

    Prioriza as features em top_features se existirem no DataFrame.
    Caso contrário, usa as colunas numéricas não-binárias com maior variância.
    Features binárias (apenas 0/1) são excluídas — causam erros no Evidently.

    Args:
        reference   : DataFrame de referência
        top_features: lista de features preferidas
        max_cols    : máximo de colunas no relatório

    Returns:
        lista de nomes de colunas selecionadas
    """
    preferred = [c for c in top_features if c in reference.columns]
    if len(preferred) >= 5:
        return preferred

    # Fallback: seleciona features numéricas com mais de 2 valores únicos
    numeric_cols = reference.select_dtypes(include=[np.number]).columns.tolist()
    valid = [
        c for c in numeric_cols
        if c != 'target'
        and reference[c].nunique() > 2
        and reference[c].std() > 0
    ]
    # Ordena por variância decrescente — features mais informativas primeiro
    variances = reference[valid].var().sort_values(ascending=False)
    return list(variances.head(max_cols).index)


def simulate_production_data(reference: pd.DataFrame,
                              n_samples: int = 200,
                              drift_intensity: float = 0.3) -> pd.DataFrame:
    """Simula dados de produção com drift moderado para demonstração.

    Em produção real, estes dados viriam dos logs da API.

    Args:
        reference      : DataFrame de referência (dados de treino)
        n_samples      : número de amostras a simular
        drift_intensity: intensidade do drift (0 = sem drift, 1 = drift severo)

    Returns:
        DataFrame com dados simulados de produção
    """
    np.random.seed(123)
    production = reference.sample(n=n_samples, replace=True).copy().reset_index(drop=True)

    # Introduzir drift nas features mais relevantes
    # Simula um cenário onde o mercado imobiliário mudou após o treino:
    # casas maiores e mais novas estão sendo vendidas

    cols = select_monitor_columns(reference, TOP_FEATURES)

    # Para cada feature monitorada, introduz drift proporcional à intensidade
    for col in cols[:5]:  # drift nas 5 features mais importantes
        if col in production.columns:
            shift = production[col].std() * drift_intensity * 0.5
            production[col] = production[col] + shift + np.random.normal(0, shift * 0.1, n_samples)

    return production


def generate_drift_report(reference: pd.DataFrame,
                           production: pd.DataFrame) -> dict:
    """Gera o relatório de data drift e salva como HTML.

    Args:
        reference  : dados de referência (treino)
        production : dados de produção (simulados ou reais)

    Returns:
        dict com métricas principais do relatório
    """
    cols = select_monitor_columns(reference, TOP_FEATURES)
    print(f'  Monitorando {len(cols)} features: '
          f'{cols[:5]}{"..." if len(cols) > 5 else ""}')

    ref_subset  = reference[cols].copy()
    prod_subset = production[cols].copy()

    # Relatório com Data Drift + Data Quality
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(reference_data=ref_subset, current_data=prod_subset)
    report.save_html(REPORT_PATH)

    # Extrair métricas do relatório para resumo no terminal
    report_dict  = report.as_dict()
    metrics_results = {}
    for metric in report_dict.get('metrics', []):
        result = metric.get('result', {})
        if 'dataset_drift' in result:
            metrics_results['dataset_drift']  = result.get('dataset_drift')
            metrics_results['drift_share']    = result.get('share_of_drifted_columns')
            metrics_results['n_drifted_cols'] = result.get('number_of_drifted_columns')

    return metrics_results


def main():
    # Verificar se os dados de referência existem
    if not os.path.exists(REFERENCE_PATH):
        raise FileNotFoundError(
            f'Arquivo não encontrado: {REFERENCE_PATH}\n'
            'Execute train.py primeiro para gerar os dados de referência.'
        )

    print('Carregando dados de referência...')
    reference = pd.read_csv(REFERENCE_PATH)
    print(f'  Referência: {reference.shape[0]:,} amostras, '
          f'{reference.shape[1]} features')

    print('Simulando dados de produção com drift...')
    production = simulate_production_data(
        reference, n_samples=200, drift_intensity=0.3
    )
    print(f'  Produção:   {production.shape[0]:,} amostras')

    print('Gerando relatório de drift...')
    metrics = generate_drift_report(reference, production)

    print()
    print('=' * 50)
    print('  RELATÓRIO DE DATA DRIFT')
    print('=' * 50)

    if metrics.get('dataset_drift') is not None:
        drift_detected = metrics['dataset_drift']
        drift_share    = metrics.get('drift_share', 0) or 0
        n_drifted      = metrics.get('n_drifted_cols', 0) or 0
        status = 'ALERTA ⚠️' if drift_detected else 'OK ✓'
        print(f'  Status             : {status}')
        print(f'  Drift detectado    : {drift_detected}')
        print(f'  Features com drift : {n_drifted} '
              f'({drift_share*100:.1f}% do total monitorado)')
    else:
        print('  Relatório gerado — abra o HTML para detalhes.')

    print(f'  Relatório salvo    : {REPORT_PATH}')
    print('=' * 50)
    print()
    print('Abra o relatório no navegador para visualização interativa:')
    print(f'  {REPORT_PATH}')


if __name__ == '__main__':
    main()
