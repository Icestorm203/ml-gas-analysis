# CatBoost + Optuna Hyperparameter Optimization

Модульная система для подбора гиперпараметров CatBoost с помощью Optuna и обучения модели на признаках TSFresh.

## 📦 Структура модуля
```
catboost_optuna/
 ├── main.py                # Точка входа
 ├── data_loader.py         # Загрузка данных
 ├── feature_extraction.py  # TSFresh признаки
 ├── objective.py           # Optuna objective-функция
 └── README.md
```

## 🧠 Возможности
- Загрузка временных рядов газов трансформатора.
- Извлечение признаков TSFresh (MinimalFCParameters).
- Оптимизация гиперпараметров CatBoost через Optuna.
- Обучение финальной модели на лучших параметрах.
- Вывод MAE на train/val.
- Финальное обучение модели на всех данных.

## 📁 Ожидаемая структура данных
```
data/
 ├── train.csv
 └── data_train/
      ├── <id1>.csv
      ├── <id2>.csv
      └── ...
```

## 🚀 Запуск
python main.py


## 📈 Результат
- Лучшие параметры CatBoost.
- MAE train / MAE val.
- Финальная модель обучена на всех данных.
