# HSE Finam Hackathon - Financial Forecasting

Решение для прогнозирования вероятностей роста акций на 1-20 дней вперед с использованием ценовых данных и новостной аналитики.

## 🎯 Задача

Предсказать вероятности роста цены акций на каждый из 20 дней вперед для каждого тикера.

**Формат выхода**: `ticker,p1,p2,p3,...,p20` где p_i - вероятность роста на день i.

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
cd final_solution
pip install -r requirements.txt
```

### 2. Запуск демо

```bash
cd final_solution
jupyter notebook demo.ipynb
```

Или используйте Python:

```python
from final_solution.solution import FinancialForecaster

# Обучение
model = FinancialForecaster()
model.fit('forecast_data/candles.csv', 'forecast_data/news.csv')
model.save('trained_model.pkl')

# Предсказание
model.predict('forecast_data/candles_2.csv', 'forecast_data/news_2.csv', 'submission.csv')
```

### 3. Проверка формата submission

```bash
cd final_solution
python test_format.py submission.csv
```

## 📁 Структура проекта

```
.
├── final_solution/          # Финальное решение
│   ├── solution.py         # Основной класс FinancialForecaster
│   ├── demo.ipynb          # Демо ноутбук с примерами
│   ├── test_format.py      # Валидатор формата submission
│   ├── requirements.txt    # Зависимости
│   └── *.md                # Документация
├── forecast_data/          # Данные (не в git)
│   ├── candles.csv        # OHLCV данные (train)
│   ├── news.csv           # Новости (train)
│   ├── candles_2.csv      # OHLCV данные (test)
│   └── news_2.csv         # Новости (test)
├── correct_submission_sample.csv  # Пример правильного формата
└── README.md              # Этот файл
```

## 📊 Архитектура решения

### Модели
- **20 CatBoost классификаторов** - по одному на каждый день (1-20)
- Каждый предсказывает вероятность роста: P(return > 0)

### Фичи (120+)
- **70+ ценовых фичей**: returns, volatility, RSI, MACD, Bollinger Bands, объемы
- **50+ новостных фичей**: sentiment, агрегации по окнам (1d, 3d, 7d, 14d, 30d)
- **Кластеризация**: 4 кластера тикеров для генерализации

### Sentiment Analysis
- Keyword-based подход (56 ключевых слов)
- Оптимизирован для русских финансовых новостей
- Быстрая обработка с кэшированием

## 📝 Формат данных

### Входные данные

**candles.csv**:
```csv
ticker,begin,open,high,low,close,volume
AFLT,2020-06-19,52.5,53.2,52.1,52.8,1000000
```

**news.csv**:
```csv
publish_date,title,publication
2020-06-19,"Заголовок","Полный текст новости"
```

### Выходные данные (submission.csv)

```csv
ticker,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20
AFLT,0.51,0.52,0.48,0.50,0.53,0.49,0.51,0.52,0.50,0.51,0.52,0.50,0.51,0.49,0.50,0.51,0.52,0.48,0.50,0.53
SBER,0.49,0.53,0.51,0.50,0.52,0.48,0.50,0.51,0.49,0.50,0.51,0.49,0.50,0.52,0.51,0.50,0.49,0.53,0.51,0.50
```

**Важно**:
- Одна строка на тикер
- p1-p20 - вероятности роста в диапазоне [0, 1]
- Без дат и returns

## 📚 Документация

Подробная документация в папке `final_solution/`:

- **[USAGE.md](final_solution/USAGE.md)** - Инструкция по использованию
- **[FORMAT_SPECIFICATION.md](final_solution/FORMAT_SPECIFICATION.md)** - Спецификация формата
- **[STATUS.md](final_solution/STATUS.md)** - Текущий статус проекта
- **[CHANGELOG.md](final_solution/CHANGELOG.md)** - История изменений

## 🔧 Требования

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
catboost>=1.2.0
```

Python 3.8+

## ⚡ Производительность

- **Обучение**: ~5-7 минут (20 моделей)
- **Предсказание**: ~30 секунд
- **Использование памяти**: ~2GB

## 🧪 Тестирование

```bash
# Проверка импорта
python -c "from final_solution.solution import FinancialForecaster; print('✅ OK')"

# Проверка формата
cd final_solution
python test_format.py submission.csv
```

## 📈 Пример использования

Полный пример в [demo.ipynb](final_solution/demo.ipynb).

Краткий пример:

```python
from final_solution.solution import FinancialForecaster

# Инициализация
model = FinancialForecaster(random_state=42)

# Обучение
model.fit(
    candles_path='forecast_data/candles.csv',
    news_path='forecast_data/news.csv'
)

# Сохранение модели
model.save('trained_model.pkl')

# Предсказание
submission = model.predict(
    candles_path='forecast_data/candles_2.csv',
    news_path='forecast_data/news_2.csv',
    output_path='submission.csv'
)

print(submission.head())
#   ticker    p1    p2    p3  ...   p20
# 0   AFLT  0.51  0.52  0.48  ...  0.53
# 1   SBER  0.49  0.53  0.51  ...  0.50
```

## 🔄 Обновления

**v2.0** (2025-10-05):
- ✅ Исправлен формат submission: `ticker,p1-p20`
- ✅ 20 классификаторов вместо 6 моделей
- ✅ Удалены LGBM и Ridge регрессоры
- ✅ Добавлена валидация формата

См. [CHANGELOG.md](final_solution/CHANGELOG.md) для деталей.

## 📄 Лицензия

HSE Finam Hackathon 2024

## 👥 Контакты

При возникновении проблем см. документацию в папке `final_solution/`.
