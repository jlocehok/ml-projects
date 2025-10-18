# Machine Learning Projects

Репозиторий с учебными проектами по машинному обучению (School 21).

## Содержание

1. **ML1 — Введение в машинное обучение**

   * Базовые понятия: supervised и unsupervised learning.
   * Классификация и регрессия.
   * Первые шаги построения модели (train/test split, метрики).

2. **ML2 — Линейная регрессия и регуляризация**

   * Аналитическое решение и градиентный спуск.
   * Ridge, Lasso, ElasticNet.
   * Bias-variance tradeoff, переобучение и недообучение.
   * Метрики качества (MAE, RMSE, R², MAPE).
   * Нормализация и полиномиальные признаки.

3. **ML3 — Оценка качества моделей**

   * Валидация: K-Fold, LOOCV, TimeSeriesSplit.
   * Подбор гиперпараметров: Grid Search, Randomized Search, Optuna.
   * Отбор признаков: Lasso, корреляция, permutation importance, SHAP.
   * Сравнение схем валидации, стабильность моделей.

4. **ML4 — Классификация**

   * Базовые алгоритмы классификации: Logistic Regression, Naive Bayes, KNN, SVM.
   * Метрики качества: Precision, Recall, F1, AUC-ROC, AUC-PR, Gini.
   * Построение confusion matrix и интерпретация ошибок.
   * Реализация логистической регрессии с нуля (SGD).
   * Работа с категориальными признаками (OneHotEncoder, CountEncoder).
   * Создание нелинейных признаков и улучшение качества модели.
   * Сравнение алгоритмов и выбор оптимальной модели.

5. **ML5 — Supervised Learning. Decision Trees и ансамбли**

   * Реализация с нуля:

     * `DecisionTreeClassifier21` / `DecisionTreeRegressor21` (критерии Gini и MSE)
     * `RandomForestClassifier21` (бутстрап, усреднение вероятностей)
     * `GDBTClassifier21` (градиентный бустинг по логистической потере)
     * `ExtraTreesClassifier21` (случайные пороги без бутстрапа)
   * Предобработка: `SimpleImputer`, `OneHotEncoder`, `CountEncoder`, `MissingIndicator`.
   * Сплит по времени (`PurchDate`) на train/val/test.
   * Оптимизация гиперпараметров через **Optuna**.
   * Сравнение с библиотеками:

     * `sklearn`: DecisionTree, RandomForest, GradientBoosting
     * `CatBoost`, `LightGBM`, `XGBoost (DART)`
   * Метрика: **Gini = 2*AUC − 1**.
   * Лучшие модели:

     * CatBoost и XGBoost (DART) с Gini(val) ≈ 0.494, Gini(test) ≈ 0.469.
   * ExtraTrees (custom) достиг целевого качества Gini ≥ 0.12.


