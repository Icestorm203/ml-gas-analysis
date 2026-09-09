# ML Gas Analysis — Диагностика трансформаторов по газам

Проект для анализа газов трансформаторного масла, построения карт Шухарта, прогнозирования выхода за UCL и гибридной диагностики состояния оборудования с помощью CatBoost и TSFresh.

## 📦 Структура проекта

ml-gas-analysis/
│
├── data/
│   ├── train.csv                 # Целевая таблица
│   └── data_train/               # Временные ряды газов
│       ├── <id1>.csv
│       ├── <id2>.csv
│       └── ...
│
├── src/
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
├── .gitignore
├── README.md                     # Главный README
└── requirements.txt

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
Перейти в любую папку, например:
cd src/shewhart_charts
И запустить:
python main.py

---

## 📁 Данные

data/train.csv
data/data_train/


---

## 🔧 Установка зависимостей

pip install -r requirements.txt

---

## 🧠 Технологии

- Python 3.10  
- TSFresh  
- CatBoost  
- Optuna  
- Scikit-learn  
- Matplotlib  
- NumPy / Pandas  