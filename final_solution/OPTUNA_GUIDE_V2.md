# 🔧 Гайд по оптимизации гиперпараметров v2.0 - РАСШИРЕННЫЙ ПОИСК

## 🆕 Что нового в v2.0?

**Значительно расширены диапазоны и параметры для каждой модели!**

### LGBM Aggressive (13+ параметров)
- `n_estimators`: 100-2000 (было 300-1000)
- `learning_rate`: 0.005-0.2 (было 0.01-0.1)
- `max_depth`: 3-15 (было 5-12)
- `num_leaves`: 15-150 (было 20-100)
- **НОВОЕ**: `min_child_samples`, `subsample_freq`, `min_split_gain`
- **НОВОЕ**: `boosting_type` (gbdt/dart) с dart-специфичными параметрами

### LGBM Conservative (14+ параметров)
- `n_estimators`: 100-1200 (было 200-700)
- `learning_rate`: 0.001-0.1 (было 0.005-0.05)
- `reg_alpha/lambda`: 0.01-20.0 (было 0.5-5.0)
- **НОВОЕ**: `min_child_weight`, `max_bin`, расширенные диапазоны

### CatBoost (12+ параметров)
- `iterations`: 200-3000 (было 500-1500)
- `learning_rate`: 0.005-0.3 (было 0.01-0.1)
- `depth`: 3-12 (было 4-10)
- `l2_leaf_reg`: 0.1-30.0 (было 1.0-10.0)
- **НОВОЕ**: `border_count`, `bagging_temperature`, `random_strength`
- **НОВОЕ**: `grow_policy` (SymmetricTree/Depthwise/Lossguide)
- **НОВОЕ**: `bootstrap_type` (Bayesian/Bernoulli/MVS)

### Ridge/Lasso/ElasticNet (расширен!)
- **НОВОЕ**: Выбор между Ridge, Lasso, ElasticNet
- **НОВОЕ**: Выбор scaler (Standard/Robust/MinMax)
- **НОВОЕ**: Различные solvers для Ridge
- **НОВОЕ**: ElasticNet с l1_ratio
- `alpha`: 1e-3 - 100.0 (было 0.01-10.0)

## Быстрый старт

### 1. Глубокая оптимизация (рекомендуется)

```bash
# 150 trials для максимального качества (~30-45 мин на модель)
python optimize_hyperparams.py --n-trials 150 --model lgbm_aggressive

# 200 trials для CatBoost (~1-1.5 часа)
python optimize_hyperparams.py --n-trials 200 --model catboost

# 300 trials для Ridge/Lasso/ElasticNet (~20-30 мин)
python optimize_hyperparams.py --n-trials 300 --model ridge
```

### 2. Сверхглубокая оптимизация

```bash
# Все модели с 200 trials (~4-6 часов)
python optimize_hyperparams.py --n-trials 200 --model all
```

### 3. Быстрая проверка

```bash
# 50 trials для быстрой проверки (~10-15 мин)
python optimize_hyperparams.py --n-trials 50 --model lgbm_aggressive
```

## Полный список оптимизируемых параметров

### LGBM Aggressive (13+ параметров)

| Параметр | Диапазон | Тип |
|----------|----------|-----|
| `n_estimators` | 100-2000 | int |
| `learning_rate` | 0.005-0.2 | float (log) |
| `max_depth` | 3-15 | int |
| `num_leaves` | 15-150 | int |
| `min_child_samples` | 5-100 | int |
| `subsample` | 0.5-1.0 | float |
| `subsample_freq` | 0-10 | int |
| `colsample_bytree` | 0.5-1.0 | float |
| `reg_alpha` | 1e-4 - 10.0 | float (log) |
| `reg_lambda` | 1e-4 - 10.0 | float (log) |
| `min_split_gain` | 0.0-1.0 | float |
| `boosting_type` | gbdt, dart | categorical |
| **Если dart**: | | |
| `drop_rate` | 0.01-0.3 | float |
| `skip_drop` | 0.3-0.7 | float |

### LGBM Conservative (14 параметров)

| Параметр | Диапазон | Тип |
|----------|----------|-----|
| `n_estimators` | 100-1200 | int |
| `learning_rate` | 0.001-0.1 | float (log) |
| `max_depth` | 2-10 | int |
| `num_leaves` | 7-63 | int |
| `min_child_samples` | 10-200 | int |
| `min_child_weight` | 1e-5 - 1e-1 | float (log) |
| `subsample` | 0.4-0.9 | float |
| `subsample_freq` | 0-7 | int |
| `colsample_bytree` | 0.4-0.9 | float |
| `reg_alpha` | 0.01-20.0 | float (log) |
| `reg_lambda` | 0.01-20.0 | float (log) |
| `min_split_gain` | 0.0-2.0 | float |
| `max_bin` | 127-511 | int |

### CatBoost (до 12 параметров)

| Параметр | Диапазон | Тип |
|----------|----------|-----|
| `iterations` | 200-3000 | int |
| `learning_rate` | 0.005-0.3 | float (log) |
| `depth` | 3-12 | int |
| `l2_leaf_reg` | 0.1-30.0 | float (log) |
| `border_count` | 32-255 | int |
| `bagging_temperature` | 0.0-10.0 | float |
| `random_strength` | 0.0-10.0 | float |
| `min_data_in_leaf` | 1-100 | int |
| `grow_policy` | SymmetricTree, Depthwise, Lossguide | categorical |
| `bootstrap_type` | Bayesian, Bernoulli, MVS | categorical |
| **Если Bernoulli**: | | |
| `subsample` | 0.5-1.0 | float |
| **Если Lossguide**: | | |
| `max_leaves` | 16-64 | int |

### Ridge/Lasso/ElasticNet (до 6 параметров)

| Параметр | Диапазон | Тип |
|----------|----------|-----|
| `model_type` | ridge, lasso, elasticnet | categorical |
| `scaler_type` | standard, robust, minmax | categorical |
| **Ridge**: | | |
| `alpha` | 1e-3 - 100.0 | float (log) |
| `solver` | auto, svd, cholesky, lsqr, sag, saga | categorical |
| **Lasso**: | | |
| `alpha` | 1e-4 - 10.0 | float (log) |
| `max_iter` | 1000-10000 | int |
| `selection` | cyclic, random | categorical |
| **ElasticNet**: | | |
| `alpha` | 1e-4 - 10.0 | float (log) |
| `l1_ratio` | 0.0-1.0 | float |
| `max_iter` | 1000-10000 | int |
| `selection` | cyclic, random | categorical |

## Рекомендованные стратегии оптимизации

### Стратегия 1: Быстрая (2-3 часа)
```bash
python optimize_hyperparams.py --n-trials 50 --model all --horizon 5d
```
- Для всех моделей
- Хорошо для первой итерации
- Найдет приличные параметры

### Стратегия 2: Сбалансированная (6-8 часов)
```bash
# Для каждой модели отдельно
python optimize_hyperparams.py --n-trials 150 --model lgbm_aggressive --horizon 5d
python optimize_hyperparams.py --n-trials 150 --model lgbm_conservative --horizon 5d
python optimize_hyperparams.py --n-trials 200 --model catboost --horizon 5d
python optimize_hyperparams.py --n-trials 300 --model ridge --horizon 5d
```
- Более глубокий поиск
- Параллельно можно запустить
- Хорошее соотношение время/качество

### Стратегия 3: Максимальная (12-24 часа)
```bash
# Для важных горизонтов
for horizon in 1d 5d 10d 20d; do
  python optimize_hyperparams.py --n-trials 200 --model all --horizon $horizon
done
```
- Оптимизация для каждого горизонта
- Максимальное качество
- Для финального submission

### Стратегия 4: Параллельная (быстрее всего)
```bash
# Запустите в разных терминалах одновременно
python optimize_hyperparams.py --n-trials 150 --model lgbm_aggressive &
python optimize_hyperparams.py --n-trials 150 --model lgbm_conservative &
python optimize_hyperparams.py --n-trials 200 --model catboost &
python optimize_hyperparams.py --n-trials 300 --model ridge &
```
- Все модели параллельно
- Экономия времени в 4 раза
- Требует достаточно CPU

## Время выполнения (обновлено)

| Модель | n_trials | Время | Параметров |
|--------|----------|-------|------------|
| Ridge/Lasso/EN | 300 | ~20-30 мин | 6 |
| LGBM Aggressive | 150 | ~30-45 мин | 13+ |
| LGBM Conservative | 150 | ~25-40 мин | 14 |
| CatBoost | 200 | ~60-90 мин | 12+ |
| Все (all) | 150 | ~2-3 часа | 45+ |
| Все (all) | 300 | ~4-6 часов | 45+ |

## Примеры результатов

### LGBM Aggressive с DART
```
✓ Лучший MAE: 0.014523
✓ Лучшие параметры:
  n_estimators: 1450
  learning_rate: 0.0234
  max_depth: 11
  num_leaves: 89
  min_child_samples: 23
  subsample: 0.87
  subsample_freq: 3
  colsample_bytree: 0.78
  reg_alpha: 0.089
  reg_lambda: 0.234
  min_split_gain: 0.012
  boosting_type: dart
  drop_rate: 0.12
  skip_drop: 0.45
```

### CatBoost с Lossguide
```
✓ Лучший MAE: 0.014102
✓ Лучшие параметры:
  iterations: 2100
  learning_rate: 0.0189
  depth: 9
  l2_leaf_reg: 4.56
  border_count: 178
  bagging_temperature: 3.2
  random_strength: 4.8
  min_data_in_leaf: 34
  grow_policy: Lossguide
  max_leaves: 48
  bootstrap_type: Bayesian
```

### ElasticNet
```
✓ Лучший MAE: 0.015234
✓ Лучшие параметры:
  model_type: elasticnet
  scaler_type: robust
  alpha: 0.234
  l1_ratio: 0.67
  max_iter: 5600
  selection: random
```

## Использование результатов

После оптимизации обновите `solution.py`:

```python
self.model_params = {
    'lgbm_aggressive': {
        'n_estimators': 1450,
        'learning_rate': 0.0234,
        'max_depth': 11,
        'num_leaves': 89,
        'min_child_samples': 23,
        'subsample': 0.87,
        'subsample_freq': 3,
        'colsample_bytree': 0.78,
        'reg_alpha': 0.089,
        'reg_lambda': 0.234,
        'min_split_gain': 0.012,
        'boosting_type': 'dart',
        'drop_rate': 0.12,
        'skip_drop': 0.45,
        'random_state': 42,
        'verbose': -1
    },
    # ... и т.д.
}
```

## Советы для максимального качества

1. **Начните с 50-100 trials** для понимания диапазона
2. **Затем 150-300 trials** для финального поиска
3. **Используйте разные горизонты** - параметры могут отличаться
4. **Параллелизация** - запускайте модели одновременно
5. **Мониторинг** - смотрите на промежуточные результаты
6. **Сохранение** - результаты автоматически в txt файлах

## Troubleshooting

### Долго работает
- Уменьшите `iterations` для CatBoost в коде (200-1500 вместо 200-3000)
- Используйте меньше trials (50 вместо 150)

### Out of memory
- Уменьшите `max_depth` и `num_leaves` диапазоны
- Закройте другие программы
- Запускайте модели по очереди, а не параллельно

---

**Версия**: 2.0 (Extended Search)
**Дата**: 2025-10-05
**Параметров**: 45+ (было 21)
