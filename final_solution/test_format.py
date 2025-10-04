"""
Скрипт для проверки формата submission
"""
import pandas as pd
import sys

def check_submission_format(filepath: str) -> bool:
    """Проверяет соответствие формату ticker,p1,p2,...,p20"""
    try:
        df = pd.read_csv(filepath)

        # Проверка колонок
        required_cols = ['ticker'] + [f'p{i}' for i in range(1, 21)]

        if list(df.columns) != required_cols:
            print(f"❌ Неправильные колонки!")
            print(f"   Ожидалось: {required_cols[:5]}...{required_cols[-3:]}")
            print(f"   Получено: {list(df.columns)[:5]}...{list(df.columns)[-3:]}")
            return False

        # Проверка уникальности тикеров
        if df['ticker'].duplicated().any():
            print(f"❌ Найдены дубликаты тикеров!")
            duplicates = df[df['ticker'].duplicated()]['ticker'].unique()
            print(f"   Дубликаты: {duplicates[:5]}")
            return False

        # Проверка диапазона вероятностей
        prob_cols = [f'p{i}' for i in range(1, 21)]
        for col in prob_cols:
            if df[col].min() < 0 or df[col].max() > 1:
                print(f"❌ Вероятности вне диапазона [0, 1] в колонке {col}")
                print(f"   Min: {df[col].min()}, Max: {df[col].max()}")
                return False

        print(f"✅ Формат submission корректен!")
        print(f"   Тикеров: {len(df)}")
        print(f"   Колонок: {len(df.columns)} (ticker + p1-p20)")
        print(f"   Пример первой строки:")
        print(f"   {df.iloc[0]['ticker']}: p1={df.iloc[0]['p1']:.3f}, p2={df.iloc[0]['p2']:.3f}, ..., p20={df.iloc[0]['p20']:.3f}")

        return True

    except Exception as e:
        print(f"❌ Ошибка при чтении файла: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_format.py <submission.csv>")
        sys.exit(1)

    filepath = sys.argv[1]
    is_valid = check_submission_format(filepath)
    sys.exit(0 if is_valid else 1)
