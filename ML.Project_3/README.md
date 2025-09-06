# ML3 — Кросс-валидация, отбор признаков и оптимизация гиперпараметров

## Описание
Учебный проект (School 21).  
Цель — научиться реализовывать свои функции для сплита и кросс-валидации, разбирать методы отбора признаков и сравнивать их, а также проводить оптимизацию гиперпараметров моделей (Grid Search, Random Search, Optuna).

## Основные темы
* Сплиты:
  * `train_test_split`, `train_val_test_split`  
  * Разделение по дате (`train_test_date_split`, `train_val_test_date_split`)  
  * Детерминированность через `random_state`  
* Кросс-валидация:
  * KFold, GroupKFold, StratifiedKFold, TimeSeriesSplit (свои реализации + sklearn)  
  * Сравнение схем, выбор оптимальной  
* Отбор признаков:
  * Lasso (L1-регуляризация)  
  * Pearson, Chi²  
  * Permutation importance  
  * SHAP values  
* Оптимизация гиперпараметров:
  * Grid Search  
  * Random Search  
  * Optuna (с TPE-сэмплером)  
  * Optuna + TimeSeriesSplit  

## Данные 
Чтобы воспроизвести результаты:  
1. Скачайте `train.json` и `test.json` с Kaggle - https://www.kaggle.com/competitions/two-sigma-connect-rental-listing-inquiries/data.  
2. Положите файлы в папку `data/` 
3. Для быстрой демонстрации в репозитории есть `sample_train.json` (500 строк) — он показывает структуру данных, но не подходит для обучения моделей.

## Практическая часть
* Датасет аренды жилья (RentHop): `train.json`, `test.json`  
* Предобработка:
  * удаление выбросов по цене (1% и 99% квантили)  
  * кодирование `interest_level` (low/medium/high)  
  * обработка `features` → бинарные фичи  
  * финальный набор: топ-20 фич + `bathrooms`, `bedrooms`, `created`  
* Эксперименты:
  * Сравнение схем кросс-валидации  
  * Отбор признаков разными методами  
  * Подбор гиперпараметров ElasticNet  

## Результаты
* Лучшая схема валидации — **TimeSeriesSplit по признаку `created`** (без утечек во времени).  
* Отбор признаков:
  * Быстрее всего — **Lasso**  
  * Лучшая интерпретируемость — **SHAP**  
* Оптимизация гиперпараметров:
  * Быстрее всего — **RandomSearch**  
  * Более стабильный результат — **Optuna**  
* Финальная модель (ElasticNet + Optuna + TimeSeriesSplit):  
  * **MAE ≈ 712**  
  * **RMSE ≈ 1120**  
  * **R² ≈ 0.78**  

## Стек
* Python 3  
* pandas, numpy  
* scikit-learn  
* optuna, shap  
* matplotlib, seaborn  

## Структура проекта
```
ML.Project_3/
├─ data/ # train.json, test.json
├─ notebooks/
│ └─ ml1_linear.ipynb
├─ README.md
```

