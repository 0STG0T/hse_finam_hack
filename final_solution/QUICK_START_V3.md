# 🚀 БЫСТРЫЙ СТАРТ

## ⚠️ КРИТИЧЕСКИ ВАЖНО

**p1-p20 = ДОХОДНОСТИ (returns), НЕ вероятности!**

Формат submission:
```csv
ticker,p1,p2,p3,...,p20
AFLT,0.0123,-0.0089,0.0234,...,-0.0112
SBER,-0.0067,0.0234,0.0089,...,0.0045
```

## 📦 Установка

```bash
pip install -r requirements.txt
```

## 🎯 Использование - demo.ipynb

```bash
jupyter notebook demo.ipynb
```

Ноутбук включает валидацию формата!

## ✅ Проверка формата

```bash
python test_returns_format.py submission.csv
python validate_final_setup.py
```

## 🏗️ Архитектура

**160 моделей** - для submission используются регрессоры!

## 🎯 Интерпретация

p1 = 0.0123 → +1.23% через 1 день
