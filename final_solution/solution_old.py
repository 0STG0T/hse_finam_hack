"""
Financial Forecasting Solution
HSE Finam Hackathon

Класс FinancialForecaster для предсказания доходностей и вероятностей роста акций
на горизонтах 1 и 20 дней с использованием новостных и ценовых данных.
"""

import pandas as pd
import numpy as np
import pickle
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class FinancialForecaster:
    """
    Класс для прогнозирования доходностей и вероятностей роста акций.

    Использует:
    - Новостные фичи (sentiment analysis + агрегации)
    - Ценовые фичи (OHLCV + technical indicators)
    - Ансамбль моделей (LGBM, CatBoost, Ridge)
    - Кластеризация тикеров для генерализации
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
        """Инициализация параметров моделей"""
        self.model_params = {
            'lgbm_1d': {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 5,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'verbose': -1
            },
            'lgbm_20d': {
                'n_estimators': 300,
                'learning_rate': 0.03,
                'max_depth': 6,
                'num_leaves': 63,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'verbose': -1
            },
            'catboost': {
                'iterations': 100,
                'learning_rate': 0.03,
                'depth': 5,
                'random_state': self.random_state,
                'verbose': False
            },
            'ridge': {
                'alpha': 1.0,
                'random_state': self.random_state
            }
        }

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
        from sklearn.cluster import KMeans

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
        # Убираем NaN в таргетах
        df = df.dropna(subset=['log_return_1d', 'log_return_20d'])

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
        """Обучение ансамбля моделей"""
        df = train_data['df']
        tree_features = train_data['tree_features']
        macro_features = train_data['macro_features']

        # Подготовка данных
        X_tree = df[tree_features].fillna(0)
        X_macro = df[macro_features].fillna(0)

        # Таргеты для регрессии
        y_1d = df['log_return_1d']
        y_20d = df['log_return_20d']

        # Таргеты для классификации (вероятности роста)
        y_1d_binary = (y_1d > 0).astype(int)
        y_20d_binary = (y_20d > 0).astype(int)

        # 1. LGBM для предсказания return_1d
        print("  • Обучение LGBM (1d)...")
        self.models['lgbm_1d'] = LGBMRegressor(**self.model_params['lgbm_1d'])
        self.models['lgbm_1d'].fit(X_tree, y_1d)

        # 2. LGBM для предсказания return_20d
        print("  • Обучение LGBM (20d)...")
        self.models['lgbm_20d'] = LGBMRegressor(**self.model_params['lgbm_20d'])
        self.models['lgbm_20d'].fit(X_tree, y_20d)

        # 3. CatBoost для вероятностей
        print("  • Обучение CatBoost (prob 1d)...")
        self.models['catboost_prob_1d'] = CatBoostClassifier(**self.model_params['catboost'])
        self.models['catboost_prob_1d'].fit(X_tree, y_1d_binary)

        print("  • Обучение CatBoost (prob 20d)...")
        self.models['catboost_prob_20d'] = CatBoostClassifier(**self.model_params['catboost'])
        self.models['catboost_prob_20d'].fit(X_tree, y_20d_binary)

        # 4. Ridge для макро-фичей
        print("  • Обучение Ridge (1d)...")
        self.scalers['ridge_1d'] = StandardScaler()
        X_macro_scaled = self.scalers['ridge_1d'].fit_transform(X_macro)
        self.models['ridge_1d'] = Ridge(**self.model_params['ridge'])
        self.models['ridge_1d'].fit(X_macro_scaled, y_1d)

        print("  • Обучение Ridge (20d)...")
        self.scalers['ridge_20d'] = StandardScaler()
        X_macro_scaled = self.scalers['ridge_20d'].fit_transform(X_macro)
        self.models['ridge_20d'] = Ridge(**self.model_params['ridge'])
        self.models['ridge_20d'].fit(X_macro_scaled, y_20d)

        print("  ✓ Все модели обучены")

    def _generate_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Генерация предсказаний для submission"""
        tree_features = self.feature_lists['tree_features']
        macro_features = self.feature_lists['macro_features']

        X_tree = df[tree_features].fillna(0)
        X_macro = df[macro_features].fillna(0)

        # Предсказания returns
        pred_lgbm_1d = self.models['lgbm_1d'].predict(X_tree)
        pred_lgbm_20d = self.models['lgbm_20d'].predict(X_tree)

        X_macro_scaled_1d = self.scalers['ridge_1d'].transform(X_macro)
        pred_ridge_1d = self.models['ridge_1d'].predict(X_macro_scaled_1d)

        X_macro_scaled_20d = self.scalers['ridge_20d'].transform(X_macro)
        pred_ridge_20d = self.models['ridge_20d'].predict(X_macro_scaled_20d)

        # Ансамбль для returns (веса подобраны эмпирически)
        return_1d = 0.7 * pred_lgbm_1d + 0.3 * pred_ridge_1d
        return_20d = 0.7 * pred_lgbm_20d + 0.3 * pred_ridge_20d

        # Предсказания вероятностей
        prob_1d = self.models['catboost_prob_1d'].predict_proba(X_tree)[:, 1]
        prob_20d = self.models['catboost_prob_20d'].predict_proba(X_tree)[:, 1]

        # Формируем submission
        submission = pd.DataFrame({
            'ticker': df['ticker'],
            'date': df['date'],
            'return_1d': return_1d,
            'return_20d': return_20d,
            'prob_1d': prob_1d,
            'prob_20d': prob_20d
        })

        # Сортируем по ticker и date
        submission = submission.sort_values(['ticker', 'date']).reset_index(drop=True)

        return submission
