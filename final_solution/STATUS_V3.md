# ✅ Статус: МАКСИМАЛЬНЫЙ АНСАМБЛЬ ГОТОВ (v3.0)

## ГЛАВНОЕ ИЗМЕНЕНИЕ

Вернули ВСЕ модели из main.ipynb для максимального скора!

### Было (v2.0):
- 20 CatBoost классификаторов
- Быстро, но не максимальное качество

### Стало (v3.0):
- **160 моделей** из main.ipynb
- **Для каждого из 20 дней**:
  * 2 LGBM (aggressive + conservative)
  * 1 CatBoost  
  * 1 Ridge (на macro features)
  * Всего 8 объектов на горизонт
- **Оптимизация весов** через scipy.optimize
- **МАКСИМАЛЬНОЕ качество**

## Архитектура ансамбля

### На КАЖДЫЙ из 20 дней обучаются:

1. **LGBM Aggressive**: n_est=750, depth=8, lr=0.05
2. **LGBM Conservative**: n_est=500, depth=5, lr=0.01
3. **CatBoost**: iterations=1000, l2=5
4. **Ridge** (macro features): alpha=1.0

Каждая модель - это регрессор + классификатор (кроме Ridge).

### Prediction:

- **Вероятности p1-p20**: Усреднение 3 классификаторов
  * (LGBM_agg_clf + LGBM_con_clf + CatBoost_clf) / 3

## Производительность

- ⏱️ **Обучение**: ~15-25 минут (160 моделей)
- 💾 **Размер**: ~50-100 MB pkl файл
- 🎯 **Качество**: МАКСИМАЛЬНОЕ (как в main.ipynb)

## Быстрый старт

```python
from solution import FinancialForecaster

# Обучение (долго, но качественно!)
model = FinancialForecaster()
model.fit('forecast_data/candles.csv', 'forecast_data/news.csv')
model.save('model_v3_full.pkl')

# Предсказание
model.predict('forecast_data/candles_2.csv', 'forecast_data/news_2.csv', 'submission.csv')
```

## Feature Sets

- **Tree features** (~70): Для LGBM и CatBoost
- **Macro features** (~16): Для Ridge
  * market_*, rank, beta, vs_market, dow_*, month, news_*, sentiment_*, cluster_*

---

**Версия**: v3.0 FULL ENSEMBLE  
**Дата**: 2025-10-05
