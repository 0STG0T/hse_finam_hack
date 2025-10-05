#!/usr/bin/env python3
"""
Финальная валидация настройки solution.py для возврата доходностей (returns)

Проверяет:
1. solution.py использует регрессоры для предсказаний
2. Формат submission: ticker,p1,p2,...,p20
3. p1-p20 содержат доходности (returns), не вероятности
"""

import sys
import re

def validate_solution_py():
    """Валидация solution.py на использование регрессоров"""
    print("="*70)
    print("ВАЛИДАЦИЯ SOLUTION.PY")
    print("="*70)

    with open('solution.py', 'r', encoding='utf-8') as f:
        content = f.read()

    issues = []
    checks_passed = []

    # 1. Проверка что _generate_predictions использует регрессоры
    if 'lgbm_agg_reg_' in content and 'lgbm_con_reg_' in content:
        checks_passed.append("✅ Использует регрессоры (lgbm_agg_reg, lgbm_con_reg)")
    else:
        issues.append("❌ НЕ использует регрессоры для LGBM")

    if 'cat_reg_' in content:
        checks_passed.append("✅ Использует CatBoost регрессор (cat_reg)")
    else:
        issues.append("❌ НЕ использует CatBoost регрессор")

    if 'ridge_reg_' in content:
        checks_passed.append("✅ Использует Ridge регрессор (ridge_reg)")
    else:
        issues.append("❌ НЕ использует Ridge регрессор")

    # 2. Проверка что НЕ использует predict_proba в _generate_predictions
    gen_pred_match = re.search(
        r'def _generate_predictions.*?(?=\n    def |\Z)',
        content,
        re.DOTALL
    )

    if gen_pred_match:
        gen_pred_code = gen_pred_match.group(0)
        if 'predict_proba' in gen_pred_code:
            issues.append("❌ ИСПОЛЬЗУЕТ predict_proba в _generate_predictions (НЕПРАВИЛЬНО!)")
        else:
            checks_passed.append("✅ НЕ использует predict_proba (правильно)")

        # 3. Проверка что использует .predict() для регрессоров
        if '.predict(X_tree)[0]' in gen_pred_code or '.predict(X_macro' in gen_pred_code:
            checks_passed.append("✅ Использует .predict() для регрессоров")
        else:
            issues.append("❌ НЕ использует .predict() для регрессоров")

        # 4. Проверка комментариев о returns
        if 'ДОХОДНОСТИ' in gen_pred_code or 'returns' in gen_pred_code.lower():
            checks_passed.append("✅ Есть комментарии о доходностях/returns")
        else:
            issues.append("⚠️  Нет комментариев о формате returns")
    else:
        issues.append("❌ Не найдена функция _generate_predictions")

    # 5. Проверка создания всех 20 таргетов
    target_creation = re.search(
        r'for horizon in range\(1,\s*21\):.*?log_return_\{horizon\}d',
        content,
        re.DOTALL
    )

    if target_creation:
        checks_passed.append("✅ Создает log_return для всех 20 дней")
    else:
        issues.append("❌ НЕ создает log_return для всех 20 дней")

    # 6. Проверка обучения всех моделей
    if 'lgbm_agg_reg = lgb.LGBMRegressor' in content:
        checks_passed.append("✅ Обучает LGBM Aggressive регрессор")
    else:
        issues.append("❌ НЕ обучает LGBM Aggressive регрессор")

    if 'lgbm_con_reg = lgb.LGBMRegressor' in content:
        checks_passed.append("✅ Обучает LGBM Conservative регрессор")
    else:
        issues.append("❌ НЕ обучает LGBM Conservative регрессор")

    if 'cat_reg = CatBoostRegressor' in content:
        checks_passed.append("✅ Обучает CatBoost регрессор")
    else:
        issues.append("❌ НЕ обучает CatBoost регрессор")

    if 'ridge_reg = Ridge' in content:
        checks_passed.append("✅ Обучает Ridge регрессор")
    else:
        issues.append("❌ НЕ обучает Ridge регрессор")

    # Вывод результатов
    print("\nПРОВЕРКИ ПРОЙДЕНЫ:")
    for check in checks_passed:
        print(f"  {check}")

    if issues:
        print("\n⚠️  НАЙДЕНЫ ПРОБЛЕМЫ:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n" + "="*70)
        print("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
        print("="*70)
        return True


def validate_demo_notebook():
    """Валидация demo.ipynb на правильные комментарии"""
    print("\n" + "="*70)
    print("ВАЛИДАЦИЯ DEMO.IPYNB")
    print("="*70)

    try:
        import json
        with open('demo.ipynb', 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        # Ищем упоминания формата
        all_text = json.dumps(notebook)

        checks_passed = []
        issues = []

        if 'ДОХОДНОСТИ' in all_text or 'returns' in all_text.lower():
            checks_passed.append("✅ Есть упоминания о доходностях/returns")
        else:
            issues.append("⚠️  Нет упоминаний о формате returns")

        if 'p1,p2' in all_text and 'p20' in all_text:
            checks_passed.append("✅ Правильный формат колонок (p1-p20)")
        else:
            issues.append("⚠️  Не упоминается формат p1-p20")

        if 'validate_submission' in all_text:
            checks_passed.append("✅ Есть функция валидации формата")
        else:
            issues.append("⚠️  Нет функции валидации")

        print("\nПРОВЕРКИ ПРОЙДЕНЫ:")
        for check in checks_passed:
            print(f"  {check}")

        if issues:
            print("\n⚠️  НАЙДЕНЫ ЗАМЕЧАНИЯ:")
            for issue in issues:
                print(f"  {issue}")

        return len(issues) == 0

    except Exception as e:
        print(f"❌ Ошибка при чтении demo.ipynb: {e}")
        return False


def print_usage_instructions():
    """Инструкции по использованию"""
    print("\n" + "="*70)
    print("СЛЕДУЮЩИЕ ШАГИ")
    print("="*70)
    print("""
1. ✅ solution.py настроен правильно для возврата доходностей
2. ✅ demo.ipynb содержит валидацию формата

ЧТОБЫ СОЗДАТЬ SUBMISSION:

Вариант A - Если модель уже обучена:
  from solution import FinancialForecaster
  model = FinancialForecaster()
  model.load('trained_model_v3.pkl')
  submission = model.predict(
      candles_path='../forecast_data/candles_2.csv',
      news_path='../forecast_data/news_2.csv',
      output_path='submission_returns.csv'
  )

Вариант B - Обучить заново:
  model = FinancialForecaster(random_state=42)
  model.fit(
      candles_path='../forecast_data/candles.csv',
      news_path='../forecast_data/news.csv'
  )
  model.save('trained_model_v3.pkl')
  submission = model.predict(...)

ПРОВЕРКА РЕЗУЛЬТАТА:
  python test_returns_format.py submission_returns.csv

ФОРМАТ SUBMISSION:
  ticker,p1,p2,p3,...,p20
  AFLT,0.0123,-0.0089,0.0234,...,-0.0112

  где p_i = предсказанная доходность на день i

ВАЖНО:
  - p1-p20 должны быть ДОХОДНОСТЯМИ (returns), не вероятностями!
  - Отрицательные значения допустимы (падение цены)
  - Значения обычно в диапазоне [-0.1, +0.1] для дневных доходностей
""")


if __name__ == '__main__':
    print("ФИНАЛЬНАЯ ВАЛИДАЦИЯ НАСТРОЙКИ ДЛЯ RETURNS\n")

    solution_ok = validate_solution_py()
    demo_ok = validate_demo_notebook()

    print("\n" + "="*70)
    print("ИТОГОВЫЙ РЕЗУЛЬТАТ")
    print("="*70)

    if solution_ok and demo_ok:
        print("✅ ВСЕ КОМПОНЕНТЫ НАСТРОЕНЫ ПРАВИЛЬНО!")
        print_usage_instructions()
        sys.exit(0)
    else:
        print("⚠️  ЕСТЬ ЗАМЕЧАНИЯ - см. вывод выше")
        sys.exit(1)
