# ML — Деревья решений и ансамбли

## Описание

Учебный проект (School 21).
Цель — реализовать свои версии деревьев и ансамблей (Decision Tree, Random Forest, Gradient Boosting, Extra Trees), построить воспроизводимый препроцессинг и сравнить с CatBoost / LightGBM / XGBoost. 

## Основные темы

* Сплит по времени: `train / validation / test` по `PurchDate`
* Предобработка:

  * числовые фичи → `SimpleImputer(median)` + `MissingIndicator`
  * категориальные: `OneHot` для low-cardinality, `CountEncoder` для high-cardinality + индикаторы пропусков
  * сборка через `ColumnTransformer`
* Реализации «с нуля»:

  * `DecisionTreeClassifier21` / `DecisionTreeRegressor21` (Gini/MSE)
  * `RandomForestClassifier21` (бутстрап, усреднение вероятностей)
  * `GDBTClassifier21` (логистический бустинг по псевдо-остаткам)
  * `ExtraTreesClassifier21` (случайные пороги без бутстрапа)
* Сравнение с библиотеками:

  * `sklearn`: `DecisionTree`, `RandomForest`, `GradientBoosting`
  * `CatBoost` (нативные категориальные, ordered boosting)
  * `LightGBM` (histogram-based, leaf-wise)
  * `XGBoost` DART (dropout деревьев) 

## Практическая часть

* Датасет: **Kaggle “Don’t Get Kicked!”** (`data/training.csv`, индекс `RefId`)
* Подход:

  1. Сплит по датам → `train/val/test`
  2. Препроцессинг числовых и категориальных фич
  3. Реализации деревьев и ансамблей с нуля
  4. Бейзлайны на `sklearn`
  5. Тюнинг CatBoost / LightGBM / XGBoost через **Optuna**
  6. Финальная оценка по **Gini = 2*AUC−1** на `val` и `test` 

## Результаты

* Одиночное дерево (custom) ≈ качество `sklearn` при одинаковых гиперпараметрах
* RandomForest (custom) > одиночного дерева
* Gradient Boosting (custom) даёт ожидаемый прирост относительно RF
* **CatBoost** и **XGBoost(DART)** — лучшие на валидации

  * CatBoost: **Gini(val) ≈ 0.494**, **Gini(test) ≈ 0.469**
* **ExtraTrees (custom)**: цель выполнена — **Gini(val) ≥ 0.12** 

## Стек

* Python 3
* numpy, pandas
* scikit-learn, category_encoders
* optuna
* lightgbm, xgboost, catboost 

## Структура проекта

```
trees_ensembles/
├─ data/
│  └─ training.csv
├─ notebooks/
│  └─ trees_ensembles.ipynb
├─ README.md
```


## Тюнинг и ньюансы

* **Optuna** с `MedianPruner`
* CatBoost: `learning_rate, iterations, depth, l2_leaf_reg`, `Pool(..., cat_features=...)`
* LightGBM: `num_leaves, min_child_samples, feature_fraction, bagging_fraction, bagging_freq, n_estimators, learning_rate`, `early_stopping(100)`
* XGBoost DART: стандартные бустинговые + `sample_type, normalize_type, rate_drop, skip_drop, one_drop`
* Проверки на переобучение: сравнение `train/val/test`, при необходимости — калибровка вероятностей


