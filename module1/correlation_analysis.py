import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

# -------------------------------
# 1. Тестові дані
# -------------------------------
# Умовні показники біохімічного аналізу крові 8 пацієнтів
data = {
    "Ліпопротеїни": [4.5, 5.0, 5.3, 5.8, 6.1, 6.4, 6.8, 7.0],
    "Гемоглобін":   [130, 135, 137, 142, 146, 149, 151, 154],
    "Глюкоза":      [4.1, 4.4, 4.6, 4.8, 5.1, 5.3, 5.5, 5.7],
    "Холестерин":   [3.9, 4.3, 4.7, 5.1, 5.4, 5.8, 6.0, 6.2]
}

df = pd.DataFrame(data)

# -------------------------------
# 2. Обчислення коефіцієнта Пірсона (між ліпопротеїнами та гемоглобіном)
# -------------------------------
r, p_value = stats.pearsonr(df["Ліпопротеїни"], df["Гемоглобін"])
print("Коефіцієнт кореляції Пірсона між ліпопротеїнами та гемоглобіном:")
print(f"r = {r:.4f}, p-value = {p_value:.6f}\n")

# -------------------------------
# 3. Побудова графіка розсіювання
# -------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(df["Ліпопротеїни"], df["Гемоглобін"], color="royalblue", label="Експериментальні дані")

# Лінія регресії
slope, intercept, _, _, _ = stats.linregress(df["Ліпопротеїни"], df["Гемоглобін"])
x_vals = np.linspace(df["Ліпопротеїни"].min(), df["Ліпопротеїни"].max(), 100)
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, color="red", label="Лінія регресії")

plt.title("Кореляція між рівнем ліпопротеїнів та гемоглобіну", fontsize=13)
plt.xlabel("Рівень ліпопротеїнів (ммоль/л)")
plt.ylabel("Рівень гемоглобіну (г/л)")
plt.text(4.6, 152, f"r = {r:.3f}", fontsize=12, color="darkred",
         bbox=dict(facecolor="white", alpha=0.7, edgecolor="darkred"))
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("correlation_scatter.png", dpi=300)
plt.show()

# -------------------------------
# 4. Матриця кореляцій (heatmap)
# -------------------------------
corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True,
            cbar_kws={"label": "Коефіцієнт кореляції"})
plt.title("Матриця кореляцій біохімічних показників", fontsize=13)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300)
plt.show()

# -------------------------------
# 5. Висновок
# -------------------------------
print("Інтерпретація результатів:")
if r > 0.9:
    relation = "дуже сильний прямий"
elif r > 0.7:
    relation = "сильний прямий"
elif r > 0.5:
    relation = "помірний прямий"
elif r > 0:
    relation = "слабкий прямий"
else:
    relation = "обернений або відсутній"

print(f"Між рівнем ліпопротеїнів і гемоглобіну спостерігається {relation} кореляційний зв’язок.")
print("З ростом ліпопротеїнів спостерігається підвищення рівня гемоглобіну.")
