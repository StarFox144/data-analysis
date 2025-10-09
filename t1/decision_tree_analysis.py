# decision_tree_analysis.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --------- параметри ----------
csv_file = "user_behavior_data.csv"   # або None -> використаємо тестовий df
target_col = "Клас_ціль"              # якщо None -> використовуємо регресію з 'Кількість_покупок'
regression_target = "Кількість_покупок"  # ім'я колонки для регресії
test_size = 0.25
random_state = 42
max_depth = 4
# ------------------------------------------

# --- дані (якщо csv не знайдено — створимо тестовий DF) ---
try:
    df = pd.read_csv(csv_file)
    print(f"Зчитано {csv_file}: {df.shape[0]} рядків, {df.shape[1]} стовпців")
except Exception as e:
    print("Не знайдено CSV — створюю тестовий датасет")
    df = pd.DataFrame({
        "Час_на_сайті_хв": [5,12,25,8,30,3,45,18,22,40],
        "Кількість_покупок": [1,2,5,1,6,0,8,3,4,7],
        "Класифікація":   [0,0,1,0,1,0,1,1,1,1],  # приклад класу
        "Вік": [23,34,45,22,39,19,50,31,29,42]
    })

# Вибір задачі: класифікація якщо target_col існує в df, і регресія інакше
if target_col in df.columns:
    problem = "classification"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
else:
    # регресія за regression_target
    problem = "regression"
    if regression_target not in df.columns:
        raise ValueError("Не знайдено колонку для регресії. Вкажи regression_target або додай клас.")
    X = df.drop(columns=[regression_target])
    y = df[regression_target]
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)

# Примітка: видалимо нечислові колонки або перетворимо, якщо треба
X = X.select_dtypes(include=[np.number]).fillna(0)

# Розбиття на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Тренування
model.fit(X_train, y_train)

# Прогноз і оцінка
y_pred = model.predict(X_test)

print("\n--- РЕЗУЛЬТАТИ ---")
if problem == "classification":
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))
else:
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))

# Дерево у текстовому вигляді
print("\n--- ДЕРЕВО (текст) ---")
print(export_text(model, feature_names=list(X.columns)))

# Важливість ознак
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n--- ВАЖЛИВІСТЬ ОЗНАК ---")
print(feat_imp)

# Візуалізація дерева
plt.figure(figsize=(12,6))
plot_tree(model, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree")
plt.tight_layout()
plt.show()
