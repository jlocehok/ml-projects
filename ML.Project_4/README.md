# ML4 — Классификация и подбор гиперпараметров

## Описание
Учебный проект (School 21).  
Цель — научиться строить и сравнивать классификационные модели, реализовать свои версии алгоритмов (LogReg, Naive Bayes, KNN), работать с признаками и оптимизировать гиперпараметры для увеличения качества.

## Основные темы
* Сплит по времени: train / validation / test  
* Предобработка данных:
  * числовые признаки (импутация, нормализация, `MissingIndicator`)  
  * категориальные признаки: OneHot (low-cardinality), CountEncoder (high-cardinality)  
* Модели:
  * **Logistic Regression**  
  * **Gaussian Naive Bayes**  
  * **KNN**  
* Реализация «с нуля»:
  * Logistic Regression (SGD, logloss, ранняя остановка)  
  * Gaussian Naive Bayes  
  * KNN (евклидово расстояние, голосование соседей)  
* Нелинейные признаки:
  * агрегаты по марке/модели (`Make_avg_cost`, `Model_avg_odo`, `Rel_odo_model`)  
  * лог-преобразования (`log_cost`, `log_odo`)  
  * новые отношения цен (`Auction_to_retail`)  
* Отбор признаков:
  * ручной порог по коэффициентам модели  
  * L1-регуляризация (Logistic Regression, solver=liblinear)  
* Оптимизация гиперпараметров:
  * **SVC (RBF kernel)** через Optuna  
  * **Logistic Regression (ElasticNet)** через Optuna (C, l1_ratio)  

## Практическая часть
* Датасет: **Kaggle "Don’t Get Kicked!" (Used Car Auction Prices)**  
* Подход:
  1. Базовые модели → LogReg, Naive Bayes, KNN  
  2. Реализация своих версий  
  3. Генерация новых фич → улучшение качества  
  4. Отбор признаков (ручной и L1)  
  5. Подбор гиперпараметров через Optuna  
  6. Финальная модель — ElasticNet LogReg  

## Результаты
* Лучшая модель: **Logistic Regression (ElasticNet)**  
* Метрики:
  * **Gini (train):** 0.514  
  * **Gini (valid):** 0.471  
  * **Gini (test):** 0.432  
* Падение качества от валидации к тесту есть, но небольшое → модель не сильно переобучена.  
* Важные гиперпараметры: **C** (сила регуляризации), **l1_ratio** (доля L1).  
* Для задачи поиска «плохих машин» (lemons) ключевая метрика — **Recall**, так как важно минимизировать FN (не пропустить дефектные автомобили).  

## Стек
* Python 3  
* pandas, numpy  
* scikit-learn  
* optuna  
* category_encoders  
* matplotlib, seaborn  

## Структура проекта
```
ML.Project_4/
├─ data/ # training.csv, test.csv
├─ notebooks/
│ └─ ML_Project_04.ipynb
├─ README.md
```

