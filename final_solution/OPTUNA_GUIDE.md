# 🔧 Гайд по оптимизации гиперпараметров с Optuna

## Что это?

`optimize_hyperparams.py` - скрипт для автоматической оптимизации гиперпараметров всех моделей ансамбля через Optuna.

## Установка

```bash
pip install optuna
```

Или:
```bash
pip install -r requirements.txt
```

## Быстрый старт

### 1. Оптимизация одной модели

```bash
# LGBM Aggressive (50 trials, ~15-20 минут)
python optimize_hyperparams.py --n-trials 50 --model lgbm_aggressive

# CatBoost (30 trials, ~20-30 минут)
python optimize_hyperparams.py --n-trials 30 --model catboost

# Ridge (100 trials, ~5 минут - быстрая модель)
python optimize_hyperparams.py --n-trials 100 --model ridge
```

### 2. Оптимизация всех моделей

```bash
# Все модели сразу (50 trials каждая, ~1-2 часа)
python optimize_hyperparams.py --n-trials 50 --model all
```

### 3. Оптимизация для другого горизонта

```bash
# Оптимизация для 10-дневного горизонта
python optimize_hyperparams.py --n-trials 50 --model lgbm_aggressive --horizon 10d

# Оптимизация для 20-дневного горизонта
python optimize_hyperparams.py --n-trials 30 --model catboost --horizon 20d
```

## Параметры

| Параметр | Описание | Default |
|----------|----------|---------|
| `--n-trials` | Количество trials для Optuna | 50 |
| `--model` | Модель для оптимизации | all |
| `--horizon` | Горизонт предсказания | 5d |
| `--candles` | Путь к candles.csv | ../forecast_data/candles.csv |
| `--news` | Путь к news.csv | ../forecast_data/news.csv |

## Оптимизируемые параметры

### LGBM Aggressive
- `n_estimators`: 300-1000
- `learning_rate`: 0.01-0.1 (log scale)
- `max_depth`: 5-12
- `num_leaves`: 20-100
- `subsample`: 0.7-0.95
- `colsample_bytree`: 0.7-0.95
- `reg_alpha`: 0.01-1.0 (log scale)
- `reg_lambda`: 0.01-1.0 (log scale)

### LGBM Conservative
- `n_estimators`: 200-700
- `learning_rate`: 0.005-0.05 (log scale)
- `max_depth`: 3-7
- `num_leaves`: 8-31
- `subsample`: 0.6-0.8
- `colsample_bytree`: 0.6-0.8
- `reg_alpha`: 0.5-5.0 (log scale)
- `reg_lambda`: 0.5-5.0 (log scale)

### CatBoost
- `iterations`: 500-1500
- `learning_rate`: 0.01-0.1 (log scale)
- `depth`: 4-10
- `l2_leaf_reg`: 1.0-10.0

### Ridge
- `alpha`: 0.01-10.0 (log scale)

## Примеры вывода

```
==============================================================
ОПТИМИЗАЦИЯ: lgbm_aggressive (horizon=5d)
==============================================================
Train: 18950, Val: 4730

[I 2025-10-05 03:30:15,123] Trial 0 finished with value: 0.016234
[I 2025-10-05 03:31:42,456] Trial 1 finished with value: 0.015987
[I 2025-10-05 03:33:11,789] Trial 2 finished with value: 0.015654
...
[I 2025-10-05 04:15:30,012] Trial 49 finished with value: 0.015123

✓ Лучший MAE: 0.014987
✓ Лучшие параметры:
  n_estimators: 850
  learning_rate: 0.0423
  max_depth: 9
  num_leaves: 67
  subsample: 0.88
  colsample_bytree: 0.82
  reg_alpha: 0.15
  reg_lambda: 0.23
```

## Использование результатов

После оптимизации обновите параметры в `solution.py`:

```python
# В методе _init_model_params()
self.model_params = {
    'lgbm_aggressive': {
        'n_estimators': 850,      # из Optuna
        'learning_rate': 0.0423,  # из Optuna
        'max_depth': 9,           # из Optuna
        'num_leaves': 67,         # из Optuna
        'subsample': 0.88,        # из Optuna
        'colsample_bytree': 0.82, # из Optuna
        'reg_alpha': 0.15,        # из Optuna
        'reg_lambda': 0.23,       # из Optuna
        'random_state': 42,
        'verbose': -1
    },
    # ... остальные модели
}
```

## Рекомендации

### Быстрая оптимизация (1-2 часа)
```bash
python optimize_hyperparams.py --n-trials 20 --model all --horizon 5d
```

### Средняя оптимизация (3-4 часа)
```bash
python optimize_hyperparams.py --n-trials 50 --model all --horizon 5d
```

### Глубокая оптимизация (6-8 часов)
```bash
# Для каждого важного горизонта
for horizon in 1d 5d 10d 20d; do
    python optimize_hyperparams.py --n-trials 100 --model all --horizon $horizon
done
```

### Советы

1. **Начните с малого**: 10-20 trials для первого прогона
2. **Фокус на важных горизонтах**: 1d, 5d, 10d, 20d
3. **Параллелизация**: Запустите несколько процессов для разных моделей
4. **Мониторинг**: Optuna показывает progress bar и промежуточные результаты
5. **Сохранение**: Результаты сохраняются в `optuna_results_*.txt`

## Продвинутые примеры

### Оптимизация с кастомными данными
```bash
python optimize_hyperparams.py \
  --n-trials 100 \
  --model lgbm_aggressive \
  --horizon 5d \
  --candles /path/to/my/candles.csv \
  --news /path/to/my/news.csv
```

### Batch оптимизация для всех горизонтов
```python
# create_optuna_batch.sh
#!/bin/bash
for model in lgbm_aggressive lgbm_conservative catboost ridge; do
  for horizon in 1d 2d 5d 10d 15d 20d; do
    python optimize_hyperparams.py \
      --n-trials 50 \
      --model $model \
      --horizon $horizon &
  done
  wait  # Ждем завершения текущей модели
done
```

## Время выполнения

| Модель | n_trials | Время (approx) |
|--------|----------|----------------|
| Ridge | 100 | ~5-10 мин |
| LGBM Aggressive | 50 | ~15-25 мин |
| LGBM Conservative | 50 | ~10-20 мин |
| CatBoost | 30 | ~20-35 мин |
| Все (all) | 50 | ~60-90 мин |

## Troubleshooting

### Ошибка: "ModuleNotFoundError: No module named 'optuna'"
```bash
pip install optuna
```

### Ошибка: "No module named 'solution'"
Убедитесь что вы в папке `final_solution/`:
```bash
cd final_solution
python optimize_hyperparams.py --n-trials 10 --model ridge
```

### Слишком долго работает
Уменьшите количество trials:
```bash
python optimize_hyperparams.py --n-trials 10 --model lgbm_aggressive
```

## Визуализация результатов

Optuna поддерживает визуализацию (требует дополнительные пакеты):

```python
import optuna

# Загрузить study из SQLite
study = optuna.load_study(
    study_name='lgbm_aggressive_5d',
    storage='sqlite:///optuna_study.db'
)

# Визуализация
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
optuna.visualization.plot_parallel_coordinate(study)
```

---

**Автор**: Generated with Claude Code
**Дата**: 2025-10-05
**Версия**: 1.0
