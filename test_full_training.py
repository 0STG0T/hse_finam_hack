#!/usr/bin/env python
"""
Тест полного обучения v3.0 FULL ENSEMBLE
"""
import sys
import os

# Убедимся что используем актуальную версию
if 'solution' in sys.modules:
    del sys.modules['solution']

sys.path.insert(0, 'final_solution')

from solution import FinancialForecaster
import pandas as pd

print("="*60)
print("ТЕСТ v3.0 FULL ENSEMBLE (160 models)")
print("="*60)

# Инициализация
model = FinancialForecaster()
print(f"\n✓ Инициализация: {len(model.model_params)} конфигураций моделей")
print(f"  Модели: {list(model.model_params.keys())}")

# Обучение
print("\n" + "="*60)
print("ЗАПУСК ОБУЧЕНИЯ...")
print("="*60)

model.fit(
    candles_path='forecast_data/candles.csv',
    news_path='forecast_data/news.csv'
)

# Проверка
total_models = len([k for k in model.models.keys()])
print(f"\n✓ ОБУЧЕНО МОДЕЛЕЙ: {total_models}")
print(f"✓ ENSEMBLE WEIGHTS: {len(model.ensemble_weights)} горизонтов")

# Сохранение
model.save('trained_model_v3_test.pkl')
print(f"\n✓ Модель сохранена: trained_model_v3_test.pkl")

# Предсказание
print("\n" + "="*60)
print("ЗАПУСК ПРЕДСКАЗАНИЯ...")
print("="*60)

submission = model.predict(
    candles_path='forecast_data/candles_2.csv',
    news_path='forecast_data/news_2.csv',
    output_path='submission_v3_test.csv'
)

print(f"\n✓ Submission создан:")
print(submission.head(3))
print(f"\n✓ Форма: {submission.shape}")
print(f"✓ Колонки: {list(submission.columns)[:5]}...{list(submission.columns)[-3:]}")

print("\n" + "="*60)
print("✓ ТЕСТ ЗАВЕРШЕН УСПЕШНО!")
print("="*60)
