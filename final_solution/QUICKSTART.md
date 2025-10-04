# Quick Start Guide

## 🚀 За 3 шага к работающему решению

### Шаг 1: Установка зависимостей

```bash
cd final_solution
pip install -r requirements.txt
```

### Шаг 2: Запуск demo ноутбука

```bash
jupyter notebook demo.ipynb
```

Выполните все ячейки последовательно.

### Шаг 3: Или используйте Python скрипт

```python
from solution import FinancialForecaster

# Создание и обучение
model = FinancialForecaster(random_state=42)
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

print(f"✓ Создан submission с {len(submission)} строками")
```

---

## 📋 Минимальный пример

```python
from solution import FinancialForecaster

model = FinancialForecaster()
model.fit('train_candles.csv', 'train_news.csv')
model.predict('test_candles.csv', 'test_news.csv', 'submission.csv')
```

Вот и всё! 🎉

---

## 📝 Формат входных данных

### candles.csv
```
ticker,begin,open,high,low,close,volume
AFLT,2020-06-19,52.5,53.2,52.1,52.8,1000000
```

### news.csv
```
publish_date,title,publication
2020-06-19,"Заголовок","Текст новости"
```

### submission.csv (выход)
```
ticker,date,return_1d,return_20d,prob_1d,prob_20d
AFLT,2024-09-09,-0.029,0.058,0.001,0.998
```

---

## 🔧 Расширенное использование

### Сохранение модели

```python
model.fit('train_candles.csv', 'train_news.csv')
model.save('my_model.pkl')
```

### Загрузка и использование

```python
model = FinancialForecaster()
model.load('my_model.pkl')
model.predict('test_candles.csv', 'test_news.csv')
```

---

## ❓ Проблемы?

1. Проверьте формат данных (см. примеры выше)
2. Убедитесь что все зависимости установлены: `pip install -r requirements.txt`
3. Откройте [README.md](README.md) для подробной документации

---

## ✨ Что делает модель?

1. **Анализирует новости** - извлекает sentiment из заголовков и текста
2. **Создает 100+ фич** - технические индикаторы, market-level агрегации
3. **Кластеризует тикеры** - для генерализации на новые компании
4. **Обучает ансамбль** - LGBM + CatBoost + Ridge
5. **Предсказывает** - доходности и вероятности на 1 и 20 дней

Готово к production! 🚀
