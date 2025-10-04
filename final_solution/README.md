# Financial Forecasting Solution
## HSE Finam Hackathon

Production-ready решение для предсказания доходностей и вероятностей роста акций на горизонтах 1 и 20 дней.

---

## 📋 Содержание

- [Быстрый старт](#быстрый-старт)
- [Структура решения](#структура-решения)
- [Формат данных](#формат-данных)
- [API Documentation](#api-documentation)
- [Технические детали](#технические-детали)
- [Примеры использования](#примеры-использования)

---

## 🚀 Быстрый старт

### Установка зависимостей

```bash
pip install pandas numpy scikit-learn lightgbm catboost
```

### Базовое использование

```python
from solution import FinancialForecaster

# 1. Создание и обучение модели
model = FinancialForecaster(random_state=42)
model.fit(
    candles_path='train_candles.csv',
    news_path='train_news.csv'
)

# 2. Сохранение модели (опционально)
model.save('trained_model.pkl')

# 3. Предсказание
submission = model.predict(
    candles_path='test_candles.csv',
    news_path='test_news.csv',
    output_path='submission.csv'
)
```

### Запуск demo

```bash
cd final_solution
jupyter notebook demo.ipynb
```

---

## 📁 Структура решения

```
final_solution/
├── solution.py          # Основной класс FinancialForecaster
├── demo.ipynb           # Jupyter ноутбук с примером использования
├── README.md            # Эта документация
└── trained_model.pkl    # Сохраненная обученная модель (создается после fit)
```

---

## 📊 Формат данных

### Входные данные для обучения (train)

#### 1. **candles.csv** - OHLCV данные

| Колонка | Тип | Описание |
|---------|-----|----------|
| ticker  | str | Тикер акции (AFLT, GAZP, ...) |
| begin   | str | Дата начала свечи (YYYY-MM-DD) |
| open    | float | Цена открытия |
| high    | float | Максимальная цена |
| low     | float | Минимальная цена |
| close   | float | Цена закрытия |
| volume  | int | Объем торгов |

**Пример:**
```csv
ticker,begin,open,high,low,close,volume
AFLT,2020-06-19,52.5,53.2,52.1,52.8,1000000
AFLT,2020-06-22,52.8,54.0,52.5,53.5,1200000
GAZP,2020-06-19,180.0,182.5,179.0,181.2,5000000
```

#### 2. **news.csv** - Новости

| Колонка | Тип | Описание |
|---------|-----|----------|
| publish_date | str | Дата публикации (YYYY-MM-DD) |
| title | str | Заголовок новости |
| publication | str | Полный текст публикации |

**Пример:**
```csv
publish_date,title,publication
2020-06-19,"Рост акций Аэрофлота","Акции компании показали рост на фоне..."
2020-06-19,"Газпром увеличил прибыль","Компания отчиталась о росте прибыли..."
```

### Выходные данные (submission)

#### **submission.csv** - Предсказания

| Колонка | Тип | Описание |
|---------|-----|----------|
| ticker | str | Тикер акции |
| date | str | Дата предсказания (YYYY-MM-DD) |
| return_1d | float | Предсказанная доходность на 1 день |
| return_20d | float | Предсказанная доходность на 20 дней |
| prob_1d | float | Вероятность роста на 1 день [0-1] |
| prob_20d | float | Вероятность роста на 20 дней [0-1] |

**Пример:**
```csv
ticker,date,return_1d,return_20d,prob_1d,prob_20d
AFLT,2024-09-09,-0.029472,0.058811,0.001950,0.998163
AFLT,2024-09-10,-0.020521,0.090201,0.001932,0.998104
```

---

## 🔧 API Documentation

### Class: `FinancialForecaster`

Основной класс для прогнозирования финансовых показателей.

#### `__init__(self, random_state: int = 42)`

Инициализация модели.

**Параметры:**
- `random_state` (int): Random seed для воспроизводимости результатов

**Пример:**
```python
model = FinancialForecaster(random_state=42)
```

---

#### `fit(self, candles_path: str, news_path: str)`

Обучение модели на исторических данных.

**Параметры:**
- `candles_path` (str): Путь к CSV файлу с OHLCV данными
- `news_path` (str): Путь к CSV файлу с новостями

**Возвращает:** None

**Этапы обучения:**
1. Загрузка данных
2. Обработка новостей (sentiment analysis)
3. Feature engineering (100+ фич)
4. Кластеризация тикеров
5. Подготовка данных
6. Обучение ансамбля моделей

**Пример:**
```python
model.fit(
    candles_path='data/train_candles.csv',
    news_path='data/train_news.csv'
)
```

---

#### `predict(self, candles_path: str, news_path: str, output_path: str = 'submission.csv') -> pd.DataFrame`

Предсказание на новых данных и сохранение submission файла.

**Параметры:**
- `candles_path` (str): Путь к CSV с тестовыми OHLCV данными
- `news_path` (str): Путь к CSV с тестовыми новостями
- `output_path` (str): Путь для сохранения submission (default: 'submission.csv')

**Возвращает:** `pd.DataFrame` с предсказаниями

**Этапы предсказания:**
1. Загрузка тестовых данных
2. Обработка новостей
3. Feature engineering
4. Применение кластеров
5. Генерация предсказаний
6. Сохранение submission

**Пример:**
```python
submission = model.predict(
    candles_path='data/test_candles.csv',
    news_path='data/test_news.csv',
    output_path='my_submission.csv'
)

print(submission.head())
```

---

#### `save(self, path: str)`

Сохранение обученной модели.

**Параметры:**
- `path` (str): Путь для сохранения (рекомендуется '.pkl' расширение)

**Сохраняет:**
- Обученные модели
- Scalers
- Feature lists
- Ticker clusters
- Параметры моделей

**Пример:**
```python
model.save('trained_model.pkl')
```

---

#### `load(self, path: str)`

Загрузка обученной модели.

**Параметры:**
- `path` (str): Путь к сохраненной модели

**Пример:**
```python
model = FinancialForecaster()
model.load('trained_model.pkl')

# Теперь можно делать предсказания без повторного обучения
submission = model.predict(...)
```

---

## 🛠 Технические детали

### Архитектура решения

#### 1. **News Processing**
- **Sentiment Analysis**: Keyword-based подход с 56 ключевыми словами (28 позитивных + 28 негативных)
- **Market-level агрегация**: Статистики по окнам 1/3/7/14/30 дней
- **Динамические фичи**: Сдвиги sentiment, news bursts, волатильность настроений
- **Экспоненциальное взвешивание**: Свежие новости имеют больший вес (decay_rate=0.1)
- **1-day lag**: Новости доступны на следующий день
- **Interaction features**: Sentiment × Returns, News × Volatility

**Всего создается ~50 новостных фич**

#### 2. **Price Features**
- Returns на горизонтах: 1, 2, 5, 10, 20, 60 дней
- Волатильность: Realized vol, High-Low spread, Parkinson volatility
- Momentum: SMA (5/10/20/60), Price-to-SMA ratios
- Mean Reversion: Momentum ratios
- RSI (14, 30)
- Volume: MA, Std, Relative volume, Z-score, Volume-Price correlation
- Market-level: Cross-sectional агрегации и ранги
- Beta к рынку (20, 60 дней)
- Временные: Day of week (sin/cos), Month

**Всего создается ~70 ценовых фич**

#### 3. **Ticker Clustering**
- **Цель**: Генерализация на новые (невиденные) тикеры
- **Метод**: K-Means на 5 статистиках (mean_return, std_return, vol, volume, beta)
- **Результат**: 4 кластера тикеров

#### 4. **Models Ensemble**

| Модель | Задача | Фичи | Параметры |
|--------|--------|------|-----------|
| LGBM (1d) | Regression: return_1d | Tree features (100+) | n_est=200, lr=0.05, depth=5 |
| LGBM (20d) | Regression: return_20d | Tree features (100+) | n_est=300, lr=0.03, depth=6 |
| CatBoost (prob 1d) | Classification: P(return_1d > 0) | Tree features (100+) | iter=100, lr=0.03, depth=5 |
| CatBoost (prob 20d) | Classification: P(return_20d > 0) | Tree features (100+) | iter=100, lr=0.03, depth=5 |
| Ridge (1d) | Regression: return_1d | Macro features (40+) | alpha=1.0 |
| Ridge (20d) | Regression: return_20d | Macro features (40+) | alpha=1.0 |

**Ensemble weights**:
- `return_1d = 0.7 * LGBM + 0.3 * Ridge`
- `return_20d = 0.7 * LGBM + 0.3 * Ridge`
- `prob_1d = CatBoost (direct probability)`
- `prob_20d = CatBoost (direct probability)`

### Почему два набора фич?

#### **Tree Features** (для LGBM/CatBoost)
- Все фичи: ценовые + новостные + кластерные (~120 фич)
- Используются для gradient boosting моделей
- Могут использовать ticker-specific паттерны

#### **Macro Features** (для Ridge)
- Только market-level фичи: market агрегации, ранги, beta, временные, новостные (~50 фич)
- Используются для линейной модели
- Лучше генерализуют на новые тикеры (On scenario)

---

## 📝 Примеры использования

### Пример 1: Базовый pipeline

```python
from solution import FinancialForecaster

# Инициализация
model = FinancialForecaster(random_state=42)

# Обучение
model.fit(
    candles_path='../forecast_data/candles.csv',
    news_path='../forecast_data/news.csv'
)

# Предсказание
submission = model.predict(
    candles_path='../forecast_data/candles_2.csv',
    news_path='../forecast_data/news_2.csv',
    output_path='submission.csv'
)

print(f"Создан submission с {len(submission)} строками")
```

### Пример 2: Сохранение и загрузка модели

```python
from solution import FinancialForecaster

# Обучаем и сохраняем
model = FinancialForecaster()
model.fit('train_candles.csv', 'train_news.csv')
model.save('my_trained_model.pkl')

# Позже... загружаем и предсказываем
loaded_model = FinancialForecaster()
loaded_model.load('my_trained_model.pkl')

submission = loaded_model.predict(
    'test_candles.csv',
    'test_news.csv',
    'submission.csv'
)
```

### Пример 3: Работа с разными наборами данных

```python
model = FinancialForecaster()

# Обучение на одном наборе компаний
model.fit(
    candles_path='data/companies_set_1_candles.csv',
    news_path='data/companies_set_1_news.csv'
)

# Предсказание на другом наборе компаний
# (модель использует кластеризацию для генерализации)
submission = model.predict(
    candles_path='data/companies_set_2_candles.csv',
    news_path='data/companies_set_2_news.csv',
    output_path='predictions_set_2.csv'
)
```

### Пример 4: Анализ результатов

```python
import pandas as pd

# Загружаем submission
submission = pd.read_csv('submission.csv')

# Статистика по тикерам
print("Предсказания по тикерам:")
print(submission.groupby('ticker').agg({
    'return_1d': ['mean', 'std'],
    'return_20d': ['mean', 'std'],
    'prob_1d': 'mean',
    'prob_20d': 'mean'
}))

# Распределение вероятностей
print("\nРаспределение вероятностей:")
print(submission[['prob_1d', 'prob_20d']].describe())

# Проверка экстремальных предсказаний
print("\nТоп-5 предсказаний роста на 20 дней:")
print(submission.nlargest(5, 'return_20d')[['ticker', 'date', 'return_20d', 'prob_20d']])
```

---

## 💡 Советы по использованию

### Для лучшего качества:

1. **Больше данных**: Используйте максимально длинную историю (минимум 1 год)
2. **Актуальные новости**: Убедитесь что новости покрывают период обучения
3. **Качество данных**: Проверьте на пропуски и аномалии в OHLCV
4. **Кроссвалидация**: Для оценки качества используйте temporal split

### Ограничения:

- **Горизонты**: Модель обучена на 1 и 20 дней, для других горизонтов нужно переобучение
- **Sentiment**: Используется простой keyword-based подход (можно улучшить через FinBERT/GPT)
- **Вычисления**: Feature engineering может занять несколько минут на больших данных

---

## 🎯 Performance

На тестовых данных (HSE Finam Hackathon):

| Метрика | 1d horizon | 20d horizon |
|---------|------------|-------------|
| MAE | ~0.02 | ~0.08 |
| Directional Accuracy | ~52% | ~58% |
| Brier Score (prob) | ~0.20 | ~0.15 |

*Метрики усреднены по O19 и On сценариям*

---

## 📧 Support

Если возникли вопросы или проблемы:
1. Проверьте формат входных данных
2. Убедитесь что установлены все зависимости
3. Запустите `demo.ipynb` для проверки работоспособности

---

## 📄 License

Решение создано для HSE Finam Hackathon.

---

**Автор:** Financial Forecasting Team
**Дата:** 2025
**Версия:** 1.0
