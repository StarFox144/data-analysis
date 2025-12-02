import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t
from sklearn.metrics import r2_score

# --- 1. ВХІДНІ ДАНІ (ПРИКЛАД) ---
# X: Незалежна змінна "Стабілізована глюкоза"
X = np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5])
# Y: Залежна змінна "Гемоглобін"
Y = np.array([6.2, 6.5, 6.9, 7.3, 7.8, 8.5, 9.1, 9.8, 10.6, 11.5])
n = len(X)
k = 2 # Кількість параметрів (a, b)

# --- 2. НЕЛІНІЙНА МОДЕЛЬ (Експоненційна: Y = a * exp(b * X)) ---
def exponential_model(X, a, b):
    return a * np.exp(b * X)

# --- 3. ОБЧИСЛЕННЯ ПАРАМЕТРІВ (МЕТОД NLS) ---
try:
    initial_guess = [6.0, 0.05]
    # popt: оптимальні параметри, pcov: коваріаційна матриця
    popt, pcov = curve_fit(exponential_model, X, Y, p0=initial_guess)

    a_hat, b_hat = popt
    
    # Стандартні похибки параметрів
    perr = np.sqrt(np.diag(pcov))
    a_se, b_se = perr

    # Прогнозовані значення Y
    Y_hat = exponential_model(X, a_hat, b_hat)

except RuntimeError:
    print("Помилка: Оптимізація параметрів не вдалася.")
    exit()

# --- 4. ОЦІНКА ЯКОСТІ МОДЕЛІ ---

# 4.1. Коефіцієнт детермінації (R^2)
R_squared = r2_score(Y, Y_hat)

# 4.2. Стандартна похибка оцінки (Se)
Residuals = Y - Y_hat
SSR = np.sum(Residuals**2) # Сума квадратів залишків
Se = np.sqrt(SSR / (n - k))

# 4.3. Оцінка значущості параметрів (t-критерій)
alpha = 0.05
df = n - k # Ступені свободи
t_crit = t.ppf(1 - alpha/2, df) 
b_t_stat = b_hat / b_se

print("=" * 60)
print("  РЕЗУЛЬТАТИ ЗАВДАННЯ 3: ОБЧИСЛЕННЯ ТА ЯКІСТЬ МОДЕЛІ")
print("=" * 60)
print(f"Обрана модель: Y = a * exp(b * X)")
print(f"Оцінене рівняння: Y = {a_hat:.4f} * exp({b_hat:.4f} * X)")
print("-" * 60)
print("--- ОЦІНКА ПАРАМЕТРІВ ---")
print(f"Параметр a: {a_hat:.4f} (Ст. похибка: {a_se:.4f})")
print(f"Параметр b: {b_hat:.4f} (Ст. похибка: {b_se:.4f})")
print("-" * 60)
print("--- ОЦІНКА ЯКОСТІ ---")
print(f"Коефіцієнт детермінації (R^2): {R_squared:.4f}")
print(f"Стандартна похибка оцінки (Se): {Se:.4f}")
print(f"t-статистика для b: {b_t_stat:.2f} (Критичне: {t_crit:.2f})")

if abs(b_t_stat) > t_crit:
    print(f"-> Параметр b статистично значущий при {alpha*100}% рівні.")
else:
    print(f"-> Параметр b НЕ є статистично значущим.")