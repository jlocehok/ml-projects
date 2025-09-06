# ML2 — Линейная регрессия и регуляризация

## Описание
Учебный проект (School 21).  
Цель — понять линейную регрессию, научиться считать метрики, применять регуляризацию (Ridge/Lasso/ElasticNet) и нормализацию признаков.

## Основные темы
* Аналитическое решение: `w = (XᵀX)⁻¹Xᵀy`  
* Линейная регрессия: аналитика, GD, SGD, mini-batch  
* Регуляризация: Ridge (L2), Lasso (L1), ElasticNet  
* Нормализация признаков: MinMax, Standard (свои реализации + sklearn)  
* Инженерия признаков: разбор текстового `features`, топ-20 бинарных фич  
* Метрики: MAE, RMSE, R²  

## Данные 
Чтобы воспроизвести результаты:  
1. Скачайте `train.json` и `test.json` с Kaggle - https://www.kaggle.com/competitions/two-sigma-connect-rental-listing-inquiries/data.  
2. Положите файлы в папку `data/` 
3. Для быстрой демонстрации в репозитории есть `sample_train.json` (500 строк) — он показывает структуру данных, но не подходит для обучения моделей.

## Практическая часть
* Датасет аренды жилья (RentHop): `train.json`, `test.json`  
* Предобработка: удаление выбросов только в train, кодирование `interest_level`  
* Модели:
  * Свои: LinearRegression (аналитика/GD/SGD/mini-batch), Ridge/Lasso/ElasticNet  
  * sklearn: LinearRegression, Ridge, Lasso, ElasticNet  
* Эксперименты:
  * Полиномиальные признаки (degree = 10)  
  * Лог-преобразование таргета (`log1p` / `expm1`)  
  * Сравнение нормализаций  

## Результаты
* Лучшая модель по RMSE и R² — **Ridge + MinMaxScaler**  
* Самая стабильная (минимальный gap train–test) — **Lasso**  
* Полиномиальные признаки без регуляризации переобучают; нормализация и регуляризация помогают  

## Стек
* Python 3  
* pandas, numpy  
* scikit-learn  
* matplotlib, seaborn  

## Структура проекта
```
ML.Project_2/
├─ data/ # train.json, test.json
├─ notebooks/
│ └─ ml1_linear.ipynb
├─ README.md
```


