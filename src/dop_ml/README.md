# DOP ML — Сравнение ML‑моделей

Модульная система для обучения и сравнения нескольких моделей машинного обучения на признаках TSFresh.

## 📦 Структура модуля
```
dop_ml/
 ├── main.py                # Точка входа
 ├── data_loader.py         # Загрузка данных
 ├── feature_extraction.py  # TSFresh признаки
 └── README.md
```

## 🧠 Модели
- RandomForestRegressor
- GradientBoostingRegressor
- XGBRegressor
- CatBoostRegressor

## 🧩 Возможности
- Загрузка временных рядов.
- Извлечение признаков TSFresh.
- Масштабирование признаков.
- Обучение 4 моделей.
- Вывод MAE train/val для каждой модели.
- Сравнение моделей в единой таблице.

## 🚀 Запуск
python main.py


## 📊 Пример вывода
```
RandomForest     | train=XX | val=YY
GradientBoosting | train=XX | val=YY
XGBoost          | train=XX | val=YY
CatBoost         | train=XX | val=YY
```