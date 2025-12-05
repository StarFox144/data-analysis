import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

# Дані часового ряду
data = np.array([1.6, 0.8, 1.2, 0.5, 0.9, 1.1, 1.1, 0.6, 1.5, 0.8, 0.9, 1.2, 0.5, 1.3, 0.8, 1.2])
N = len(data)

# --- а) та б) Побудова графіка та наближена оцінка ---
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(range(1, N + 1), data, marker='o', linestyle='-', color='blue')
plt.title('а) Графік часового ряду y(t)')
plt.xlabel('Час (t)')
plt.ylabel('y(t)')
plt.grid(True, linestyle=':', alpha=0.6)

# --- в) Графік y(t+1) від y(t) ---
y_t = data[:-1]
y_t_plus_1 = data[1:]

plt.subplot(2, 1, 2)
plt.scatter(y_t, y_t_plus_1, color='red')
plt.title('в) Графік залежності y(t+1) від y(t) (Lag Plot)')
plt.xlabel('y(t)')
plt.ylabel('y(t+1)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
print("-" * 50)

# --- в) Точний розрахунок коефіцієнта автокореляції першого порядку ---
# 1. Середнє значення
mean_y = np.mean(data)
# 2. Дисперсія (знаменник)
denominator = np.sum((data - mean_y)**2)
# 3. Автоковаріація (чисельник)
numerator = np.sum((data[:-1] - mean_y) * (data[1:] - mean_y))
# 4. Коефіцієнт автокореляції
rho_1_manual = numerator / denominator

# Перевірка за допомогою вбудованої функції (рекомендовано в реальній роботі)
# acf повертає масив, де перший елемент - для лагу 0 (завжди 1), другий - для лагу 1
rho_1_lib = acf(data, nlags=1, adjusted=False)[1] 

print(f"1. Середнє значення (ȳ): {mean_y:.4f}")
print(f"2. Автоковаріація (Чисельник): {numerator:.4f}")
print(f"3. Дисперсія (Знаменник): {denominator:.4f}")
print(f"\nКоефіцієнт автокореляції першого порядку (ρ̂₁):")
print(f"   Ручний розрахунок: {rho_1_manual:.4f}")
print(f"   Перевірка через NumPy/Statsmodels: {rho_1_lib:.4f}")

