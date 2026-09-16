# ML Gas Analysis — Диагностика трансформаторов по газам

Проект для анализа газов трансформаторного масла, построения карт Шухарта, прогнозирования выхода за UCL и гибридной диагностики состояния оборудования с помощью CatBoost и TSFresh.

## 📦 Структура проекта
```
ml-gas-analysis/
│── data/
│   ├── train.csv                 # Целевая таблица
│   └── data_train/               # Временные ряды газов
│       ├── <id1>.csv
│       ├── <id2>.csv
│       └── ...
│
│── src/
│   ├── catboost_optuna/          # Оптимизация CatBoost через Optuna
│   │   ├── main.py
│   │   ├── data_loader.py
│   │   ├── feature_extraction.py
│   │   ├── objective.py
│   │   └── README.md
│   │
│   ├── dop_ml/                   # Сравнение ML‑моделей
│   │   ├── main.py
│   │   ├── data_loader.py
│   │   ├── feature_extraction.py
│   │   └── README.md
│   │
│   ├── linear_regression/        # Линейная + логистическая регрессия
│   │   ├── main.py
│   │   └── README.md
│   │
│   ├── shewhart_catboost/        # Гибридная диагностика (Шухарт + CatBoost)
│   │   ├── main.py
│   │   └── README.md
│   │
│   ├── shewhart_charts/          # Карты Шухарта
│   │   ├── main.py
│   │   └── README.md
│   │
│   └── shewhart_forecast/        # Прогноз выхода за UCL
│       ├── main.py
│       └── README.md
│
│── .gitignore
│── README.md                     # Главный README
└── requirements.txt
```

---

## 📊 Описание модулей

### **1. catboost_optuna**
Оптимизация гиперпараметров CatBoost через Optuna.

- TSFresh признаки  
- RandomizedSearchCV + Optuna  
- MAE train/val  
- Финальная модель  

---

### **2. dop_ml**
Сравнение нескольких ML‑моделей:

- RandomForest  
- GradientBoosting  
- XGBoost  
- CatBoost  

Вывод MAE для каждой модели.

---

### **3. linear_regression**
Двухэтапная обработка:

- Логистическая регрессия → детекция выбросов  
- CatBoost → прогноз  
- Коррекция выбросов  
- Гистограмма сравнения  

---

### **4. shewhart_catboost**
Гибридная диагностика:

- Шухарт → аномалии  
- CatBoost → RUL  
- Итоговый статус объекта  

---

### **5. shewhart_charts**
Построение карт Шухарта:

- CL, LCL, UCL  
- точки вне контроля  
- PNG‑графики  
- первые 5 объектов  

---

### **6. shewhart_forecast**
Прогноз выхода за UCL:

- линейный тренд  
- шаги до пересечения  
- статус тренда  

---

## 🚀 Запуск проекта

1. Скачать проект
```
git clone https://github.com/<your-username>/ml-gas-analysis.git
cd ml-gas-analysis
```
2. Установить Python

Проект протестирован на Python 3.12.

Проверить версию:
```
python --version
```
3. Создать виртуальное окружение
```
python -m venv .venv
```
4. Активировать окружение (Windows (CMD))
```
.\.venv\Scripts\activate.bat
```
5. Установить зависимости
```
pip install -r requirements.txt
```
6. Запустить любой модуль
Пример:
```
cd src/dop_ml
python main.py
```
То же самое для остальных:

- src/catboost_optuna/main.py

- src/dop_ml/main.py

- src/linear_regression/main.py

- src/shewhart_catboost/main.py

- src/shewhart_charts/main.py

- src/shewhart_forecast/main.py

---

## 📁 Данные

data/train.csv

data/data_train/<id>.csv

---

## 🧠 Технологии

- Python 3.12
- TSFresh
- CatBoost
- Optuna
- Scikit-learn
- Matplotlib
- NumPy / Pandas