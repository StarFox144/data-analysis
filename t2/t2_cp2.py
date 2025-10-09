# --- Перевірка та встановлення lifelines ---
import subprocess
import sys

try:
    from lifelines import CoxPHFitter
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lifelines"])
    from lifelines import CoxPHFitter

# --- Імпорт бібліотек ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Генерація симуляційних даних ---
np.random.seed(42)

n = 200  # кількість користувачів
data = pd.DataFrame({
    "user_id": range(1, n + 1),
    "session_time": np.random.exponential(scale=30, size=n),  # середній час сесії (хв)
    "recommendations_used": np.random.poisson(lam=3, size=n),  # кількість рекомендацій
    "logins": np.random.randint(1, 20, size=n),  # кількість входів
    "account_age": np.random.randint(10, 365, size=n),  # вік акаунта (днів)
})

# --- Генеруємо “час до відтоку” та подію ---
data["time_to_churn"] = np.random.exponential(scale=180, size=n)  # час до події
data["churned"] = np.random.binomial(1, 0.4, size=n)  # 1 - користувач пішов, 0 - ще активний

# --- Побудова моделі Кокса ---
cph = CoxPHFitter()
cph.fit(data[["time_to_churn", "churned", "session_time", "recommendations_used", "logins", "account_age"]],
        duration_col="time_to_churn", event_col="churned")

# --- Вивід результатів ---
print("\n=== РЕЗУЛЬТАТИ МОДЕЛІ КОКСА ===")
cph.print_summary()

# --- Побудова та збереження графіка ---
plt.figure(figsize=(8, 5))
cph.plot()
plt.title("Вплив поведінкових змінних на ризик відтоку користувачів")
plt.tight_layout()
plt.savefig("cox_model_plot.png", dpi=300)
plt.show()

print("\n✅ Графік збережено як 'cox_model_plot.png' у поточній директорії.")

# --- Прогноз для конкретного користувача ---
example_user = pd.DataFrame({
    "session_time": [25],
    "recommendations_used": [2],
    "logins": [5],
    "account_age": [120]
})
predicted_risk = cph.predict_partial_hazard(example_user)
print("\nОцінений ризик відтоку користувача:", float(predicted_risk))
