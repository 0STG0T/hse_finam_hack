# 🚀 Быстрый старт - FULL ENSEMBLE v3.0

## Что это?

Максимально мощный ансамбль из **160 моделей** для прогноза вероятностей роста акций.

### Архитектура:

Для КАЖДОГО из 20 дней обучаются:
- 2 LGBM (aggressive + conservative) - регрессор + классификатор
- 1 CatBoost - регрессор + классификатор  
- 1 Ridge (macro features) - регрессор + scaler

**Итого**: 8 объектов × 20 = 160 моделей

## Установка

```bash
cd final_solution
pip install -r requirements.txt
```

## Использование

### 1. Обучение (долго, ~15-25 минут)

```python
from solution import FinancialForecaster

model = FinancialForecaster()
model.fit('../forecast_data/candles.csv', '../forecast_data/news.csv')
model.save('trained_model_v3.pkl')
```

Вывод при обучении:
```
[6/6] Обучение ансамбля моделей...
  • Обучение полного ансамбля для p1-p20...
    Модели на горизонт: 2xLGBM + CatBoost + Ridge = 4 модели
    Tree features: 75, Macro features: 50
    ✓ День 5/20: MAE=0.016005, веса=['0.45', '0.30', '0.20', '0.05']
    ✓ День 10/20: MAE=0.019463, веса=['0.40', '0.35', '0.20', '0.05']
    ...
  ✓ Обучено 160 моделей (20 горизонтов × 8 моделей)
```

### 2. Предсказание (быстро, ~30 сек)

```python
from solution import FinancialForecaster

model = FinancialForecaster()
model.load('trained_model_v3.pkl')

submission = model.predict(
    '../forecast_data/candles_2.csv',
    '../forecast_data/news_2.csv',
    'submission.csv'
)

print(submission.head())
#   ticker       p1       p2  ...      p20
# 0   AFLT  0.5123  0.5089  ...  0.4908
# 1   SBER  0.4876  0.5211  ...  0.5102
```

### 3. Проверка формата

```bash
python test_format.py submission.csv
```

Вывод:
```
✅ Формат submission корректен!
   Тикеров: 19
   Колонок: 21 (ticker + p1-p20)
```

## Что внутри?

### Модели на каждый день:

1. **LGBM Aggressive Regressor**
   - 750 деревьев, depth=8, lr=0.05
   - Агрессивный fitting для захвата сложных паттернов

2. **LGBM Aggressive Classifier**  
   - 100 деревьев, depth=5, lr=0.05
   - Для вероятностей p1-p20

3. **LGBM Conservative Regressor**
   - 500 деревьев, depth=5, lr=0.01
   - Более консервативный, меньше overfitting

4. **LGBM Conservative Classifier**
   - 100 деревьев, depth=4, lr=0.01
   - Для вероятностей p1-p20

5. **CatBoost Regressor**
   - 1000 итераций, l2_leaf_reg=5
   - Хорошо работает с категориальными фичами

6. **CatBoost Classifier**
   - 700 итераций
   - Для вероятностей p1-p20

7. **Ridge Regressor**
   - alpha=1.0, только macro features (~16)
   - Линейная модель для стабильности

8. **StandardScaler**
   - Для Ridge модели

### Feature Sets:

- **Tree features (~75)**: Все фичи для LGBM и CatBoost
  * Prices: returns, volatility, SMA, RSI, MACD, Bollinger, volumes
  * News: sentiment, counts, weighted features
  * Clusters: 4 ticker clusters

- **Macro features (~50)**: Только для Ridge
  * market_*, rank, beta, vs_market, dow_*, month
  * news_*, sentiment_*, weighted_sentiment_*
  * cluster_*

### Ансамблирование:

**Вероятности p1-p20**:
```python
p = (LGBM_agg_clf + LGBM_con_clf + CatBoost_clf) / 3
```

Усреднение 3 классификаторов для стабильных вероятностей.

## Производительность

| Метрика | Значение |
|---------|----------|
| Обучение | ~15-25 минут |
| Предсказание | ~30 секунд |
| Размер модели | ~50-100 MB |
| Количество моделей | 160 |
| Качество | МАКСИМАЛЬНОЕ |

## Troubleshooting

### Проблема: Долго обучается
**Решение**: Это нормально, 160 моделей требуют времени. Можно уменьшить:
- `n_estimators` для LGBM (750→300, 500→200)
- `iterations` для CatBoost (1000→500, 700→400)

### Проблема: Много памяти
**Решение**: 
- Закройте другие программы
- Обучайте на машине с ≥8GB RAM

### Проблема: Пропускаются дни при обучении
**Решение**: Убедитесь что создаются все log_return_1d до log_return_20d таргеты.

## Сравнение версий

| Версия | Моделей | Время | Качество |
|--------|---------|-------|----------|
| v1.0 | 6 | ~2 мин | ❌ Неправильный формат |
| v2.0 | 20 | ~5 мин | ⭐⭐⭐ Хорошо |
| v3.0 | 160 | ~20 мин | ⭐⭐⭐⭐⭐ МАКСИМУМ |

## Что дальше?

1. Обучите модель на своих данных
2. Сделайте предсказания
3. Проверьте формат через `test_format.py`
4. Submit!

---

**Версия**: v3.0 FULL ENSEMBLE  
**Дата**: 2025-10-05  
**Источник**: Восстановлено из main.ipynb
