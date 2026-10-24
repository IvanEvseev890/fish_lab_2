# Fish Weight Prediction Model: Лабораторная работа №2

![Fish](images/Fishlength.jpg)

## О проекте
Этот репозиторий содержит выполнение лабораторной работы №2 по дисциплине «Введение в искусственный интеллект». Цель — построить воспроизводимую модель регрессии для предсказания массы рыбы (`Weight`) на основе её измерений (`Species`, `Length1`, `Length2`, `Length3`, `Height`, `Width`), сравнить линейные и нелинейные подходы, а также зафиксировать эксперименты в MLflow.

## Окружение и зависимости
- **Версия Python:** 3.13
- **Основные библиотеки:** `pandas`, `numpy`, `scikit-learn`, `mlflow`, `matplotlib`, `seaborn`, `pytest`
- **Случайное зерно (random_state):** 42

### Установка зависимостей
```bash
pip install pandas numpy scikit-learn mlflow matplotlib seaborn pytest