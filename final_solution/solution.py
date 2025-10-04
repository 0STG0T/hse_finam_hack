"""
Financial Forecasting Solution
HSE Finam Hackathon

Класс FinancialForecaster для предсказания вероятностей роста акций
на горизонтах от 1 до 20 дней с использованием новостных и ценовых данных.

Формат submission: ticker,p1,p2,p3,...,p20
где p1-p20 - вероятности роста на каждый из 20 дней вперед.
"""

import pandas as pd
import numpy as np
import pickle
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML libraries
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize


class FinancialForecaster:
    """
    Класс для прогнозирования вероятностей роста акций на 1-20 дней вперед.

    Использует:
    - Новостные фичи (sentiment analysis + агрегации)
    - Ценовые фичи (OHLCV + technical indicators)
    - Мощный ансамбль для каждого из 20 дней:
        * 2 LGBM (aggressive + conservative)
        * 1 CatBoost
        * 1 Ridge (на macro features)
    - Оптимизация весов ансамбля через scipy.optimize
    - Dual feature sets: tree features для деревьев, macro для Ridge
    - Кластеризация тикеров для генерализации

    Output format: ticker,p1,p2,...,p20 (одна строка на тикер)
    """

    def __init__(self, random_state: int = 42):
        """
        Инициализация модели.

        Args:
            random_state: Random seed для воспроизводимости
        """
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_lists = {}
        self.ticker_clusters = {}
        self.news_sentiment_cache = None

        # Параметры моделей
        self._init_model_params()

    def _init_model_params(self):
        """Инициализация параметров моделей из main.ipynb"""
        self.model_params = {
            'lgbm_aggressive': {
                'n_estimators': 750,
                'learning_rate': 0.05,
                'max_depth': 8,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': self.random_state,
                'verbose': -1
            },
            'lgbm_conservative': {
                'n_estimators': 500,
                'learning_rate': 0.01,
                'max_depth': 5,
                'num_leaves': 16,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'reg_alpha': 1.0,
                'reg_lambda': 1.0,
                'random_state': self.random_state,
                'verbose': -1
            },
            'lgbm_clf_aggressive': {
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 5,
                'random_state': self.random_state,
                'verbose': -1
            },
            'lgbm_clf_conservative': {
                'n_estimators': 100,
                'learning_rate': 0.01,
                'max_depth': 4,
                'random_state': self.random_state,
                'verbose': -1
            },
            'catboost': {
                'iterations': 1000,
                'l2_leaf_reg': 5,
                'random_seed': self.random_state,
                'grow_policy': 'Depthwise',
                'verbose': 0
            },
            'catboost_clf': {
                'iterations': 700,
                'random_seed': self.random_state,
                'grow_policy': 'Depthwise',
                'verbose': 0
            },
            'ridge': {
                'alpha': 1.0,
                'random_state': self.random_state
            }
        }

        # Веса ансамбля (будут оптимизированы при обучении)
        self.ensemble_weights = {}

    def fit(self, candles_path: str, news_path: str):
        """
        Обучение модели на исторических данных.

        Args:
            candles_path: Путь к CSV с OHLCV данными
                         Columns: ticker, begin, open, high, low, close, volume
            news_path: Путь к CSV с новостями
                      Columns: publish_date, title, publication
        """
        print("=" * 60)
        print("ОБУЧЕНИЕ МОДЕЛИ")
        print("=" * 60)

        # 1. Загрузка данных
        print("\n[1/6] Загрузка данных...")
        candles = pd.read_csv(candles_path)
        news = pd.read_csv(news_path)

        candles['date'] = pd.to_datetime(candles['begin']).dt.date
        candles = candles.sort_values(['ticker', 'date']).reset_index(drop=True)

        print(f"  • Загружено {len(candles)} свечей по {candles['ticker'].nunique()} тикерам")
        print(f"  • Загружено {len(news)} новостей")

        # 2. Обработка новостей (sentiment analysis)
        print("\n[2/6] Обработка новостей (sentiment analysis)...")
        news_df = self._process_news(news)

        # 3. Feature engineering
        print("\n[3/6] Feature engineering...")
        df = self._create_features(candles, news_df)

        # 4. Кластеризация тикеров
        print("\n[4/6] Кластеризация тикеров...")
        df = self._cluster_tickers(df)

        # 5. Подготовка данных для обучения
        print("\n[5/6] Подготовка данных для обучения...")
        train_data = self._prepare_training_data(df)

        # 6. Обучение моделей
        print("\n[6/6] Обучение ансамбля моделей...")
        self._train_models(train_data)

        print("\n" + "=" * 60)
        print("✓ ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print("=" * 60)

    def predict(self, candles_path: str, news_path: str, output_path: str = 'submission.csv'):
        """
        Предсказание на новых данных и сохранение submission файла.

        Args:
            candles_path: Путь к CSV с OHLCV данными для предсказания
            news_path: Путь к CSV с новостями для предсказания
            output_path: Путь для сохранения submission файла

        Returns:
            DataFrame с предсказаниями в формате submission
        """
        print("=" * 60)
        print("ПРЕДСКАЗАНИЕ")
        print("=" * 60)

        # 1. Загрузка данных
        print("\n[1/4] Загрузка тестовых данных...")
        candles = pd.read_csv(candles_path)
        news = pd.read_csv(news_path)

        candles['date'] = pd.to_datetime(candles['begin']).dt.date
        candles = candles.sort_values(['ticker', 'date']).reset_index(drop=True)

        print(f"  • Загружено {len(candles)} свечей по {candles['ticker'].nunique()} тикерам")
        print(f"  • Период: {candles['date'].min()} - {candles['date'].max()}")

        # 2. Обработка новостей
        print("\n[2/4] Обработка новостей...")
        news_df = self._process_news(news)

        # 3. Feature engineering
        print("\n[3/4] Feature engineering...")
        df = self._create_features(candles, news_df)
        df = self._apply_ticker_clusters(df)

        # 4. Генерация предсказаний
        print("\n[4/4] Генерация предсказаний...")
        submission = self._generate_predictions(df)

        # Сохранение
        submission.to_csv(output_path, index=False)
        print(f"\n✓ Submission сохранен: {output_path}")
        print(f"  • Строк: {len(submission)}")
        print(f"  • Тикеров: {submission['ticker'].nunique()}")

        print("\n" + "=" * 60)
        print("✓ ПРЕДСКАЗАНИЕ ЗАВЕРШЕНО")
        print("=" * 60)

        return submission

    def save(self, path: str):
        """Сохранение обученной модели"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_lists': self.feature_lists,
            'ticker_clusters': self.ticker_clusters,
            'model_params': self.model_params,
            'random_state': self.random_state
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✓ Модель сохранена: {path}")

    def load(self, path: str):
        """Загрузка обученной модели"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_lists = model_data['feature_lists']
        self.ticker_clusters = model_data['ticker_clusters']
        self.model_params = model_data['model_params']
        self.random_state = model_data['random_state']

        print(f"✓ Модель загружена: {path}")

    # ============================================================
    # ПРИВАТНЫЕ МЕТОДЫ
    # ============================================================

    def _process_news(self, news: pd.DataFrame) -> pd.DataFrame:
        """Обработка новостей с sentiment analysis"""
        news_df = news.copy()

        # Keyword-based sentiment analyzer
        positive_keywords = [
            'рост', 'прибыль', 'увеличени', 'развити', 'успех', 'позитив',
            'укрепл', 'повыш', 'подъ', 'восстановл', 'улучш', 'достиж',
            'прорыв', 'эффективн', 'оптимизац', 'инновац', 'модерниз',
            'расширен', 'диверсифик', 'стабилизац', 'рекорд', 'максимум',
            'дивиденд', 'выручк', 'рентабельн', 'капитализац'
        ]

        negative_keywords = [
            'падени', 'убыт', 'снижени', 'кризис', 'риск', 'угроз',
            'дефолт', 'банкротств', 'спад', 'потер', 'сокращ', 'упад',
            'провал', 'неудач', 'проблем', 'задолженн', 'санкци',
            'штраф', 'скандал', 'коррупц', 'арест', 'минимум', 'обвал',
            'девальвац', 'инфляц', 'рецесс', 'стагнац'
        ]

        def calculate_sentiment(text):
            if pd.isna(text):
                return 0.0
            text_lower = str(text).lower()
            pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
            neg_count = sum(1 for kw in negative_keywords if kw in text_lower)
            total = pos_count + neg_count
            if total == 0:
                return 0.0
            sentiment = (pos_count - neg_count) / total
            return np.tanh(sentiment * 1.5)

        news_df['full_text'] = news_df['title'].fillna('').astype(str) + ' ' + news_df['publication'].fillna('').astype(str)
        news_df['sentiment'] = news_df['full_text'].apply(calculate_sentiment)
        news_df['date'] = pd.to_datetime(news_df['publish_date'])

        print(f"  • Средний sentiment: {news_df['sentiment'].mean():.3f}")

        return news_df

    def _create_features(self, candles: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """Создание всех фич (ценовых + новостных)"""
        df = candles.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Ценовые фичи
        df = self._create_price_features(df)

        # Новостные фичи
        df = self._create_news_features(df, news_df)

        print(f"  • Создано {df.shape[1]} фич")

        return df

    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание ценовых технических индикаторов"""
        # Возвраты
        df['returns'] = df.groupby('ticker')['close'].pct_change()
        for horizon in [1, 2, 5, 10, 20, 60]:
            df[f'return_{horizon}d'] = df.groupby('ticker')['close'].pct_change(horizon)
            df[f'log_return_{horizon}d'] = np.log(
                df.groupby('ticker')['close'].shift(-horizon) / df['close']
            )

        # Волатильность
        for window in [5, 10, 20, 60]:
            df[f'realized_vol_{window}d'] = df.groupby('ticker')['returns'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )

        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        for window in [20]:
            df[f'hl_vol_{window}d'] = df.groupby('ticker')['hl_spread'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

        df['vol_ratio_5_20'] = df['realized_vol_5d'] / (df['realized_vol_20d'] + 1e-8)
        df['vol_ratio_10_60'] = df['realized_vol_10d'] / (df['realized_vol_60d'] + 1e-8)

        # Скользящие средние
        for window in [5, 10, 20, 60]:
            df[f'sma_{window}'] = df.groupby('ticker')['close'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

        for window in [5, 20, 60]:
            df[f'price_to_sma_{window}'] = df['close'] / (df[f'sma_{window}'] + 1e-8) - 1

        df['momentum_5_20'] = (df['sma_5'] / (df['sma_20'] + 1e-8)) - 1
        df['momentum_10_60'] = (df['sma_10'] / (df['sma_60'] + 1e-8)) - 1

        # RSI
        def compute_rsi(series, period):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))

        df['rsi_14'] = df.groupby('ticker')['close'].transform(lambda x: compute_rsi(x, 14))
        df['rsi_30'] = df.groupby('ticker')['close'].transform(lambda x: compute_rsi(x, 30))

        # Объемы
        for window in [5, 20]:
            df[f'volume_ma_{window}'] = df.groupby('ticker')['volume'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'volume_std_{window}'] = df.groupby('ticker')['volume'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )

        df['volume_rel'] = df['volume'] / (df['volume_ma_20'] + 1e-8)
        df['volume_zscore'] = (df['volume'] - df['volume_ma_20']) / (df['volume_std_20'] + 1e-8)

        # Market-level фичи
        market_agg = df.groupby('date').agg({
            'returns': ['mean', 'std', 'median'],
            'realized_vol_20d': ['mean', 'std'],
            'volume_rel': ['mean']
        })
        market_agg.columns = ['_'.join(col).strip() for col in market_agg.columns.values]
        market_agg = market_agg.add_prefix('market_')
        df = df.merge(market_agg, left_on='date', right_index=True, how='left')

        # Cross-sectional ранги
        df['return_rank'] = df.groupby('date')['returns'].rank(pct=True)
        df['volume_rank'] = df.groupby('date')['volume_rel'].rank(pct=True)
        df['vol_rank'] = df.groupby('date')['realized_vol_20d'].rank(pct=True)

        # Beta
        df['beta_20'] = df.groupby('ticker').apply(
            lambda x: x['returns'].rolling(20, min_periods=1).corr(x['market_returns_mean'])
        ).reset_index(level=0, drop=True)
        df['beta_60'] = df.groupby('ticker').apply(
            lambda x: x['returns'].rolling(60, min_periods=1).corr(x['market_returns_mean'])
        ).reset_index(level=0, drop=True)

        df['return_vs_market'] = df['returns'] - df['market_returns_mean']
        df['vol_vs_market'] = df['realized_vol_20d'] - df['market_realized_vol_20d_mean']

        # Временные фичи
        df['dow'] = df['date'].dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 5)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 5)
        df['month'] = df['date'].dt.month

        return df

    def _create_news_features(self, df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """Создание новостных фич"""
        # Market-level агрегация новостей
        unique_dates = sorted(df['date'].unique())
        windows = [1, 3, 7, 14, 30]

        news_features_list = []
        for date in unique_dates:
            features = {'date': date}
            for window in windows:
                start_date = pd.to_datetime(date) - pd.Timedelta(days=window)
                mask = (news_df['date'] > start_date) & (news_df['date'] <= pd.to_datetime(date))
                window_news = news_df[mask]

                features[f'news_count_{window}d'] = len(window_news)
                if len(window_news) > 0:
                    features[f'news_sentiment_mean_{window}d'] = window_news['sentiment'].mean()
                    features[f'news_sentiment_std_{window}d'] = window_news['sentiment'].std() if len(window_news) > 1 else 0.0
                    features[f'news_sentiment_min_{window}d'] = window_news['sentiment'].min()
                    features[f'news_sentiment_max_{window}d'] = window_news['sentiment'].max()
                else:
                    features[f'news_sentiment_mean_{window}d'] = 0.0
                    features[f'news_sentiment_std_{window}d'] = 0.0
                    features[f'news_sentiment_min_{window}d'] = 0.0
                    features[f'news_sentiment_max_{window}d'] = 0.0

            news_features_list.append(features)

        news_features_df = pd.DataFrame(news_features_list)

        # Динамические характеристики
        news_features_df['sentiment_shift_3_1'] = news_features_df['news_sentiment_mean_3d'] - news_features_df['news_sentiment_mean_1d']
        news_features_df['sentiment_shift_7_3'] = news_features_df['news_sentiment_mean_7d'] - news_features_df['news_sentiment_mean_3d']
        news_features_df['news_burst_3_1'] = news_features_df['news_count_3d'] / (news_features_df['news_count_1d'] + 1e-6)
        news_features_df['news_burst_7_3'] = news_features_df['news_count_7d'] / (news_features_df['news_count_3d'] + 1e-6)

        # Weighted sentiment (экспоненциальное взвешивание по времени)
        weighted_features_list = []
        for date in unique_dates:
            features = {'date': date}
            for window in [7, 14, 30]:
                start_date = pd.to_datetime(date) - pd.Timedelta(days=window)
                mask = (news_df['date'] > start_date) & (news_df['date'] <= pd.to_datetime(date))
                window_news = news_df[mask].copy()

                if len(window_news) > 0:
                    window_news['days_ago'] = (pd.to_datetime(date) - window_news['date']).dt.days
                    window_news['weight'] = np.exp(-0.1 * window_news['days_ago'])
                    total_weight = window_news['weight'].sum()
                    if total_weight > 0:
                        features[f'weighted_sentiment_{window}d'] = (window_news['sentiment'] * window_news['weight']).sum() / total_weight
                    else:
                        features[f'weighted_sentiment_{window}d'] = 0.0
                else:
                    features[f'weighted_sentiment_{window}d'] = 0.0

            weighted_features_list.append(features)

        weighted_features_df = pd.DataFrame(weighted_features_list)
        news_features_df = news_features_df.merge(weighted_features_df, on='date', how='left')

        # 1-day lag (новости доступны на следующий день)
        news_features_df['date'] = pd.to_datetime(news_features_df['date']) + pd.Timedelta(days=1)

        # Мёрдж с основным датафреймом
        df = df.merge(news_features_df, on='date', how='left')
        news_cols = [c for c in news_features_df.columns if c != 'date']
        df[news_cols] = df[news_cols].fillna(0)

        # Interaction features (новости × цены)
        if 'return_2d' in df.columns:
            df['sentiment_return_mult_3d'] = df['news_sentiment_mean_3d'] * df['return_2d']
        if 'return_5d' in df.columns:
            df['sentiment_return_mult_7d'] = df['news_sentiment_mean_7d'] * df['return_5d']
        if 'realized_vol_5d' in df.columns:
            df['news_volume_volatility_3d'] = df['news_count_3d'] * df['realized_vol_5d']
        if 'market_returns_mean' in df.columns:
            df['weighted_sentiment_market_return_7d'] = df['weighted_sentiment_7d'] * df['market_returns_mean']

        return df

    def _cluster_tickers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Кластеризация тикеров для генерализации на новые тикеры"""

        # Статистики для кластеризации
        ticker_stats = df.groupby('ticker').agg({
            'returns': ['mean', 'std'],
            'realized_vol_20d': ['mean'],
            'volume_rel': ['mean'],
            'beta_20': ['mean']
        }).reset_index()

        ticker_stats.columns = ['ticker', 'mean_return', 'std_return', 'vol_mean', 'volume_mean', 'beta_mean']
        ticker_stats = ticker_stats.fillna(0)

        # K-means кластеризация
        X = ticker_stats[['mean_return', 'std_return', 'vol_mean', 'volume_mean', 'beta_mean']].values
        kmeans = KMeans(n_clusters=4, random_state=self.random_state, n_init=10)
        ticker_stats['cluster'] = kmeans.fit_predict(X)

        # Сохраняем маппинг тикер -> кластер
        self.ticker_clusters = dict(zip(ticker_stats['ticker'], ticker_stats['cluster']))

        # Добавляем кластерные фичи
        df['cluster'] = df['ticker'].map(self.ticker_clusters).fillna(-1).astype(int)
        for cluster_id in range(4):
            df[f'cluster_{cluster_id}'] = (df['cluster'] == cluster_id).astype(int)

        print(f"  • Создано {ticker_stats['cluster'].nunique()} кластеров тикеров")

        return df

    def _apply_ticker_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применение кластеров к новым данным"""
        df['cluster'] = df['ticker'].map(self.ticker_clusters).fillna(-1).astype(int)
        for cluster_id in range(4):
            df[f'cluster_{cluster_id}'] = (df['cluster'] == cluster_id).astype(int)
        return df

    def _prepare_training_data(self, df: pd.DataFrame) -> Dict:
        """Подготовка данных для обучения"""
        # Убираем строки где нет хотя бы одного таргета из 20
        target_cols = [f'log_return_{i}d' for i in range(1, 21) if f'log_return_{i}d' in df.columns]
        df = df.dropna(subset=target_cols, how='all')

        # Определяем фичи
        exclude_cols = ['ticker', 'date', 'begin', 'open', 'high', 'low', 'close', 'volume',
                       'returns', 'cluster'] + [c for c in df.columns if 'log_return' in c or 'return_' in c]

        tree_features = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]

        macro_features = [c for c in tree_features if
                         'market_' in c or 'rank' in c or 'beta' in c or
                         'vs_market' in c or 'dow_' in c or 'month' in c or
                         'news_' in c or 'sentiment_' in c or 'weighted_sentiment' in c or
                         'cluster_' in c]

        self.feature_lists = {
            'tree_features': tree_features,
            'macro_features': macro_features
        }

        print(f"  • Tree features: {len(tree_features)}")
        print(f"  • Macro features: {len(macro_features)}")

        return {
            'df': df,
            'tree_features': tree_features,
            'macro_features': macro_features
        }

    def _train_models(self, train_data: Dict):
        """
        Обучение полного ансамбля для предсказания вероятностей роста на 1-20 дней.
        Для каждого горизонта обучаем:
        - 2 LGBM (aggressive + conservative) - регрессоры и классификаторы
        - 1 CatBoost - регрессор и классификатор
        - 1 Ridge (только на macro features) - регрессор
        Затем оптимизируем веса ансамбля
        """
        df = train_data['df']
        tree_features = train_data['tree_features']
        macro_features = train_data['macro_features']

        print("  • Обучение полного ансамбля для p1-p20...")
        print(f"    Модели на горизонт: 2xLGBM + CatBoost + Ridge = 4 модели")
        print(f"    Tree features: {len(tree_features)}, Macro features: {len(macro_features)}")

        for horizon in range(1, 21):
            target_col = f'log_return_{horizon}d'

            if target_col not in df.columns:
                print(f"    ⚠️  Пропускаем день {horizon} - нет таргета {target_col}")
                continue

            # Подготовка данных
            y = df[target_col].copy()
            valid_mask = ~y.isna()

            # Tree features
            X_tree = df[tree_features].fillna(0)[valid_mask]
            y_valid = y[valid_mask]
            y_binary = (y_valid > 0).astype(int)

            # Macro features
            X_macro = df[macro_features].fillna(0)[valid_mask]

            if len(y_valid) < 100:
                print(f"    ⚠️  Пропускаем день {horizon} - мало данных ({len(y_valid)})")
                continue

            # === 1. LGBM Aggressive ===
            lgbm_agg_reg = lgb.LGBMRegressor(**self.model_params['lgbm_aggressive'])
            lgbm_agg_reg.fit(X_tree, y_valid)

            lgbm_agg_clf = lgb.LGBMClassifier(**self.model_params['lgbm_clf_aggressive'])
            lgbm_agg_clf.fit(X_tree, y_binary)

            # === 2. LGBM Conservative ===
            lgbm_con_reg = lgb.LGBMRegressor(**self.model_params['lgbm_conservative'])
            lgbm_con_reg.fit(X_tree, y_valid)

            lgbm_con_clf = lgb.LGBMClassifier(**self.model_params['lgbm_clf_conservative'])
            lgbm_con_clf.fit(X_tree, y_binary)

            # === 3. CatBoost ===
            cat_reg = CatBoostRegressor(**self.model_params['catboost'])
            cat_reg.fit(X_tree, y_valid)

            cat_clf = CatBoostClassifier(**self.model_params['catboost_clf'])
            cat_clf.fit(X_tree, y_binary)

            # === 4. Ridge (только macro features) ===
            ridge_reg = Ridge(**self.model_params['ridge'])

            # Стандартизация для Ridge
            scaler = StandardScaler()
            X_macro_scaled = scaler.fit_transform(X_macro)
            ridge_reg.fit(X_macro_scaled, y_valid)

            # Сохраняем модели
            self.models[f'lgbm_agg_reg_{horizon}d'] = lgbm_agg_reg
            self.models[f'lgbm_agg_clf_{horizon}d'] = lgbm_agg_clf
            self.models[f'lgbm_con_reg_{horizon}d'] = lgbm_con_reg
            self.models[f'lgbm_con_clf_{horizon}d'] = lgbm_con_clf
            self.models[f'cat_reg_{horizon}d'] = cat_reg
            self.models[f'cat_clf_{horizon}d'] = cat_clf
            self.models[f'ridge_reg_{horizon}d'] = ridge_reg
            self.scalers[f'ridge_scaler_{horizon}d'] = scaler

            # === Оптимизация весов ансамбля ===
            # Делаем предсказания всех моделей
            pred_agg = lgbm_agg_reg.predict(X_tree)
            pred_con = lgbm_con_reg.predict(X_tree)
            pred_cat = cat_reg.predict(X_tree)
            pred_ridge = ridge_reg.predict(X_macro_scaled)

            predictions_list = [pred_agg, pred_con, pred_cat, pred_ridge]

            # Оптимизируем веса
            weights = self._optimize_ensemble_weights(y_valid, predictions_list)
            self.ensemble_weights[f'{horizon}d'] = weights

            if horizon % 5 == 0:
                # Показываем метрики ансамбля
                ens_pred = sum(w * p for w, p in zip(weights, predictions_list))
                mae_ens = mean_absolute_error(y_valid, ens_pred)
                print(f"    ✓ День {horizon}/20: MAE={mae_ens:.6f}, веса={[f'{w:.2f}' for w in weights]}")

        total_models = len([k for k in self.models.keys() if any(x in k for x in ['_reg_', '_clf_'])])
        print(f"  ✓ Обучено {total_models} моделей ({total_models//8} горизонтов × 8 моделей)")

    def _optimize_ensemble_weights(self, y_true, predictions_list):
        """Оптимизация весов ансамбля для минимизации MAE"""
        def objective(weights):
            pred = sum(w * p for w, p in zip(weights, predictions_list))
            return mean_absolute_error(y_true, pred)

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(predictions_list))]
        x0 = np.ones(len(predictions_list)) / len(predictions_list)

        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    def _generate_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация предсказаний для submission в формате ticker,p1,p2,...,p20

        Использует ансамбль из 3 классификаторов для каждого горизонта:
        - LGBM Aggressive + Conservative classifiers
        - CatBoost classifier
        Усредняем вероятности с весами
        """
        tree_features = self.feature_lists['tree_features']
        macro_features = self.feature_lists['macro_features']

        # Берем последнюю дату по каждому тикеру
        last_dates = df.groupby('ticker')['date'].max().reset_index()
        last_dates.columns = ['ticker', 'max_date']

        # Фильтруем только последние даты
        df_last = df.merge(last_dates, on='ticker')
        df_last = df_last[df_last['date'] == df_last['max_date']].copy()

        predictions_list = []

        for _, row in df_last.iterrows():
            ticker = row['ticker']

            # Формируем фичи для этого тикера
            X_tree = pd.DataFrame([row[tree_features]]).fillna(0)
            X_macro = pd.DataFrame([row[macro_features]]).fillna(0)

            # Предсказываем p1-p20
            probs = {'ticker': ticker}

            for horizon in range(1, 21):
                # Проверяем наличие моделей
                lgbm_agg_key = f'lgbm_agg_clf_{horizon}d'
                lgbm_con_key = f'lgbm_con_clf_{horizon}d'
                cat_key = f'cat_clf_{horizon}d'

                if all(k in self.models for k in [lgbm_agg_key, lgbm_con_key, cat_key]):
                    # Предсказываем вероятности от классификаторов
                    prob_agg = self.models[lgbm_agg_key].predict_proba(X_tree)[:, 1][0]
                    prob_con = self.models[lgbm_con_key].predict_proba(X_tree)[:, 1][0]
                    prob_cat = self.models[cat_key].predict_proba(X_tree)[:, 1][0]

                    # Усредняем вероятности (можно использовать веса если есть)
                    # Простое усреднение 3 классификаторов
                    prob = (prob_agg + prob_con + prob_cat) / 3.0

                    probs[f'p{horizon}'] = prob
                else:
                    # Если модели не обучены, возвращаем нейтральную вероятность
                    probs[f'p{horizon}'] = 0.5

            predictions_list.append(probs)

        # Формируем финальный submission
        submission = pd.DataFrame(predictions_list)

        # Сортируем по ticker
        submission = submission.sort_values('ticker').reset_index(drop=True)

        # Убеждаемся что все колонки p1-p20 есть
        for i in range(1, 21):
            if f'p{i}' not in submission.columns:
                submission[f'p{i}'] = 0.5

        # Правильный порядок колонок: ticker,p1,p2,...,p20
        cols = ['ticker'] + [f'p{i}' for i in range(1, 21)]
        submission = submission[cols]

        return submission
