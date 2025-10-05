#!/usr/bin/env python
"""
Оптимизация гиперпараметров через Optuna для всех моделей ансамбля.

Usage:
    python optimize_hyperparams.py --n-trials 50 --model lgbm_aggressive
    python optimize_hyperparams.py --n-trials 30 --model catboost
    python optimize_hyperparams.py --n-trials 100 --model all
"""
import sys
sys.path.insert(0, '.')

import argparse
import optuna
import numpy as np
import pandas as pd
from solution import FinancialForecaster
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def load_data(candles_path='../forecast_data/candles.csv',
              news_path='../forecast_data/news.csv'):
    """Загрузка и подготовка данных"""
    print("Загрузка данных...")

    model = FinancialForecaster()

    # Загружаем и обрабатываем данные
    candles = pd.read_csv(candles_path)
    news = pd.read_csv(news_path)

    candles['date'] = pd.to_datetime(candles['begin']).dt.date
    candles = candles.sort_values(['ticker', 'date']).reset_index(drop=True)

    # Обработка новостей
    news_df = model._process_news(news)

    # Feature engineering
    df = model._create_features(candles, news_df)
    df = model._cluster_tickers(df)
    df = model._apply_ticker_clusters(df)

    # Подготовка данных для обучения
    train_data = model._prepare_training_data(df)

    print(f"✓ Загружено {len(df)} строк")
    print(f"✓ Tree features: {len(train_data['tree_features'])}")
    print(f"✓ Macro features: {len(train_data['macro_features'])}")

    return train_data


def objective_lgbm_aggressive(trial, X_train, y_train, X_val, y_val):
    """Objective для LGBM Aggressive - максимальный поиск"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'num_leaves': trial.suggest_int('num_leaves', 15, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 0, 10),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'random_state': 42,
        'verbose': -1
    }

    # Для dart добавляем дополнительные параметры
    if params['boosting_type'] == 'dart':
        params['drop_rate'] = trial.suggest_float('drop_rate', 0.01, 0.3)
        params['skip_drop'] = trial.suggest_float('skip_drop', 0.3, 0.7)

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    return mean_absolute_error(y_val, pred)


def objective_lgbm_conservative(trial, X_train, y_train, X_val, y_val):
    """Objective для LGBM Conservative - широкий поиск с упором на регуляризацию"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1200, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'num_leaves': trial.suggest_int('num_leaves', 7, 63),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1e-1, log=True),
        'subsample': trial.suggest_float('subsample', 0.4, 0.9),
        'subsample_freq': trial.suggest_int('subsample_freq', 0, 7),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 20.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 20.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 2.0),
        'max_bin': trial.suggest_int('max_bin', 127, 511),
        'random_state': 42,
        'verbose': -1
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    return mean_absolute_error(y_val, pred)


def objective_catboost(trial, X_train, y_train, X_val, y_val):
    """Objective для CatBoost - расширенный поиск"""
    params = {
        'iterations': trial.suggest_int('iterations', 200, 3000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'depth': trial.suggest_int('depth', 3, 12),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 30.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
        'random_seed': 42,
        'verbose': 0
    }

    # Дополнительные параметры для разных bootstrap
    if params['bootstrap_type'] == 'Bayesian':
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 10.0)
    elif params['bootstrap_type'] == 'Bernoulli':
        params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)

    # Параметры для Lossguide
    if params['grow_policy'] == 'Lossguide':
        params['max_leaves'] = trial.suggest_int('max_leaves', 16, 64)

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    return mean_absolute_error(y_val, pred)


def objective_ridge(trial, X_train, y_train, X_val, y_val):
    """Objective для Ridge - расширенный поиск с preprocessing"""
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.linear_model import Ridge, Lasso, ElasticNet

    # Выбор модели
    model_type = trial.suggest_categorical('model_type', ['ridge', 'lasso', 'elasticnet'])

    # Выбор scaler
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'robust', 'minmax'])

    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Создание модели
    if model_type == 'ridge':
        params = {
            'alpha': trial.suggest_float('alpha', 1e-3, 100.0, log=True),
            'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga']),
            'random_state': 42
        }
        model = Ridge(**params)
    elif model_type == 'lasso':
        params = {
            'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True),
            'max_iter': trial.suggest_int('max_iter', 1000, 10000),
            'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
            'random_state': 42
        }
        model = Lasso(**params)
    else:  # elasticnet
        params = {
            'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
            'max_iter': trial.suggest_int('max_iter', 1000, 10000),
            'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
            'random_state': 42
        }
        model = ElasticNet(**params)

    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_val_scaled)

    return mean_absolute_error(y_val, pred)


def optimize_model(model_name, train_data, n_trials=50, horizon='5d'):
    """Оптимизация конкретной модели"""
    print(f"\n{'='*60}")
    print(f"ОПТИМИЗАЦИЯ: {model_name} (horizon={horizon})")
    print(f"{'='*60}")

    df = train_data['df']
    tree_features = train_data['tree_features']
    macro_features = train_data['macro_features']

    # Выбираем фичи
    if model_name == 'ridge':
        features = macro_features
    else:
        features = tree_features

    # Готовим данные для конкретного горизонта
    target_col = f'log_return_{horizon}'
    if target_col not in df.columns:
        print(f"⚠️  Пропуск - нет таргета {target_col}")
        return None

    # Временной split (80/20)
    dates = sorted(df['date'].unique())
    split_idx = int(len(dates) * 0.8)
    train_dates = dates[:split_idx]
    val_dates = dates[split_idx:]

    train_df = df[df['date'].isin(train_dates)].copy()
    val_df = df[df['date'].isin(val_dates)].copy()

    # Убираем NaN
    train_df = train_df.dropna(subset=[target_col] + features)
    val_df = val_df.dropna(subset=[target_col] + features)

    X_train = train_df[features].fillna(0)
    y_train = train_df[target_col]
    X_val = val_df[features].fillna(0)
    y_val = val_df[target_col]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Выбираем objective функцию
    objective_map = {
        'lgbm_aggressive': objective_lgbm_aggressive,
        'lgbm_conservative': objective_lgbm_conservative,
        'catboost': objective_catboost,
        'ridge': objective_ridge
    }

    if model_name not in objective_map:
        print(f"⚠️  Неизвестная модель: {model_name}")
        return None

    # Создаем study
    study = optuna.create_study(
        direction='minimize',
        study_name=f'{model_name}_{horizon}'
    )

    # Оптимизация
    objective_func = lambda trial: objective_map[model_name](trial, X_train, y_train, X_val, y_val)

    study.optimize(objective_func, n_trials=n_trials, show_progress_bar=True)

    # Результаты
    print(f"\n✓ Лучший MAE: {study.best_value:.6f}")
    print(f"✓ Лучшие параметры:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return study


def main():
    parser = argparse.ArgumentParser(description='Оптимизация гиперпараметров через Optuna')
    parser.add_argument('--n-trials', type=int, default=100, help='Количество trials (default: 100, рекомендуется 150-300 для максимального качества)')
    parser.add_argument('--model', type=str, default='all',
                       choices=['lgbm_aggressive', 'lgbm_conservative', 'catboost', 'ridge', 'all'],
                       help='Какую модель оптимизировать')
    parser.add_argument('--horizon', type=str, default='5d',
                       help='Горизонт для оптимизации (default: 5d)')
    parser.add_argument('--candles', type=str, default='../forecast_data/candles.csv',
                       help='Путь к candles.csv')
    parser.add_argument('--news', type=str, default='../forecast_data/news.csv',
                       help='Путь к news.csv')

    args = parser.parse_args()

    print("="*60)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    print(f"Trials: {args.n_trials}")
    print(f"Model: {args.model}")
    print(f"Horizon: {args.horizon}")

    # Загружаем данные
    train_data = load_data(args.candles, args.news)

    # Оптимизируем
    models_to_optimize = ['lgbm_aggressive', 'lgbm_conservative', 'catboost', 'ridge'] if args.model == 'all' else [args.model]

    results = {}
    for model_name in models_to_optimize:
        study = optimize_model(model_name, train_data, args.n_trials, args.horizon)
        if study:
            results[model_name] = study

    # Финальный отчет
    print("\n" + "="*60)
    print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
    print("="*60)

    for model_name, study in results.items():
        print(f"\n{model_name}:")
        print(f"  Лучший MAE: {study.best_value:.6f}")
        print(f"  Параметры: {study.best_params}")

    # Сохраняем результаты
    output_file = f'optuna_results_{args.model}_{args.horizon}.txt'
    with open(output_file, 'w') as f:
        f.write("OPTUNA OPTIMIZATION RESULTS\n")
        f.write("="*60 + "\n\n")
        for model_name, study in results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Best MAE: {study.best_value:.6f}\n")
            f.write(f"  Best params:\n")
            for key, value in study.best_params.items():
                f.write(f"    {key}: {value}\n")
            f.write("\n")

    print(f"\n✓ Результаты сохранены: {output_file}")


if __name__ == '__main__':
    main()
