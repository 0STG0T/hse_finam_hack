#!/usr/bin/env python
"""
Тест что submission содержит returns (а не вероятности)
"""
import sys
sys.path.insert(0, 'final_solution')

from solution import FinancialForecaster
import pandas as pd
import numpy as np

print("="*60)
print("ТЕСТ: Проверка что p1-p20 = RETURNS (не вероятности)")
print("="*60)

# Загружаем существующий submission если есть
try:
    submission = pd.read_csv('final_solution/submission.csv')
    print(f"\n✓ Загружен submission.csv: {submission.shape}")
    
    # Проверяем диапазон значений
    print("\nПроверка диапазона значений:")
    for col in ['p1', 'p5', 'p10', 'p15', 'p20']:
        if col in submission.columns:
            min_val = submission[col].min()
            max_val = submission[col].max()
            mean_val = submission[col].mean()
            
            # Вероятности были бы в [0, 1]
            # Returns обычно в [-0.1, 0.1]
            
            is_probability = (min_val >= 0) and (max_val <= 1) and (abs(mean_val - 0.5) < 0.2)
            is_return = (min_val < 0) or (max_val > 1) or (abs(mean_val) < 0.1)
            
            status = "❌ ВЕРОЯТНОСТЬ" if is_probability else "✅ RETURN"
            print(f"  {col}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f} → {status}")
    
    # Финальный вердикт
    print("\n" + "="*60)
    all_returns = [submission[f'p{i}'].min() < 0 or submission[f'p{i}'].max() > 1 
                   for i in range(1, 21) if f'p{i}' in submission.columns]
    
    if any(all_returns):
        print("✅ SUBMISSION СОДЕРЖИТ RETURNS!")
        print("   Отрицательные значения и/или значения > 1 найдены")
    else:
        print("⚠️  ПОХОЖЕ НА ВЕРОЯТНОСТИ!")
        print("   Все значения в [0, 1] - возможно неправильный формат")
        
except FileNotFoundError:
    print("\n⚠️  submission.csv не найден")
    print("   Запустите обучение и предсказание сначала")

print("\n" + "="*60)
print("Как должны выглядеть правильные значения:")
print("="*60)
print("ticker,p1,p2,p3,p4,p5,...")
print("AFLT,0.0123,-0.0089,0.0234,0.0012,-0.0156,...")
print("SBER,-0.0067,0.0234,0.0089,0.0012,-0.0145,...")
print("\nДиапазон: обычно от -0.05 до +0.05 (-5% до +5%)")
print("Отрицательные = падение, Положительные = рост")
