# Быстрый старт

## Формат submission
```csv
ticker,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20
AFLT,0.51,0.52,0.48,0.50,0.53,0.49,0.51,0.52,0.50,0.51,0.52,0.50,0.51,0.49,0.50,0.51,0.52,0.48,0.50,0.53
SBER,0.49,0.53,0.51,0.50,0.52,0.48,0.50,0.51,0.49,0.50,0.51,0.49,0.50,0.52,0.51,0.50,0.49,0.53,0.51,0.50
```

**p1-p20**: Вероятности роста цены на каждый из 20 дней вперед
**Одна строка на тикер** (не по датам!)

## Использование

### 1. Обучение модели
```python
from solution import FinancialForecaster

model = FinancialForecaster()
model.fit('candles.csv', 'news.csv')
model.save('trained_model.pkl')
```

### 2. Предсказание
```python
from solution import FinancialForecaster

model = FinancialForecaster()
model.load('trained_model.pkl')
model.predict('test_candles.csv', 'test_news.csv', 'submission.csv')
```

### 3. Полный пайплайн
```python
from solution import FinancialForecaster

# Обучение
model = FinancialForecaster()
model.fit('candles.csv', 'news.csv')

# Предсказание
submission = model.predict('test_candles.csv', 'test_news.csv', 'submission.csv')

print(submission.head())
# Output:
#   ticker    p1    p2    p3  ...   p20
# 0   AFLT  0.51  0.52  0.48  ...  0.53
# 1   SBER  0.49  0.53  0.51  ...  0.50
```

## Архитектура

### Модели
- **20 CatBoost классификаторов** - по одному на каждый день (1-20)
- Каждый классификатор предсказывает P(return > 0) для своего горизонта

### Фичи
- **70+ ценовых фичей**: OHLCV, returns, volatility, technical indicators
- **50+ новостных фичей**: sentiment, агрегации по окнам (1d, 3d, 7d, 14d, 30d)
- **Кластеризация тикеров**: 4 кластера для генерализации на новые тикеры

### Sentiment Analysis
- Keyword-based подход (56 ключевых слов)
- Быстро и эффективно для русских финансовых новостей
- Кэширование обработанных новостей

## Требования
```bash
pip install pandas numpy scikit-learn catboost
```

## Примечания
- Для каждого тикера берется **последняя доступная дата**
- На этой дате делаются предсказания p1-p20
- Submission содержит одну строку на тикер
