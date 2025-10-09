# timeseries_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox 

# ============================================================
# 1. Завантаження даних
# ============================================================
csv_filename = 'user_behavior_data.csv'
try:
    df = pd.read_csv(csv_filename, index_col='Дата', parse_dates=True)
    ts = df['Показник_Активності']
    print("="*70)
    print(f"✅ 1-2. Дані успішно завантажено з файлу: {csv_filename}")
    print("="*70)
except FileNotFoundError:
    print(f"ПОМИЛКА: Файл '{csv_filename}' не знайдено. Спочатку запустіть generate_timeseries_data.py")
    exit()

# ============================================================
# 3. Зобразити дані графічно (Висновок про наявність тренду)
# ============================================================

plt.figure(figsize=(12, 5))
ts.plot(label='Часовий ряд')
# Додавання ковзного середнього для візуалізації тренду
ts.rolling(window=12).mean().plot(color='red', linestyle='-', label='Ковзне середнє (Тренд)')
plt.title('3. Часовий ряд та Візуалізація Тренду')
plt.xlabel('Дата')
plt.ylabel('Показник Активності')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("\n--- Візуальний висновок ---")
print("Графік показує наявність зростаючого тренду та сезонних коливань.")

# ============================================================
# 4. Критерій наявності тенденції (Використання Ljung-Box Test)
# ============================================================
# H0: Автокореляція відсутня (ряд є випадковим, тренд відсутній).
# Якщо ряд має тренд, то кореляція на малих лагах (наприклад, lag=1) буде високою.

# Використовуємо Ljung-Box Test для ряду
ljung_box_trend_test = acorr_ljungbox(ts, lags=[1], return_df=True)
p_value_trend = ljung_box_trend_test['lb_pvalue'].iloc[0]
alpha = 0.05

print("\n" + "="*70)
print("4. Критерій наявності тенденції (Ljung-Box Test на автокореляцію)")
print("="*70)
print(f"P-значення Ljung-Box Test (Lag 1): {p_value_trend:.4f}")

if p_value_trend < alpha:
    print(f"\nВисновок: P-value < 0.05. Нульова гіпотеза про випадковість відхиляється.")
    print("📢 ТЕНДЕНЦІЯ В ЧАСОВОМУ РЯДУ ПРИСУТНЯ (ряд не є випадковим/незалежним).")
else:
    print(f"\nВисновок: P-value ≥ 0.05. Нульова гіпотеза про випадковість не відхиляється.")
    print("📢 ТЕНДЕНЦІЯ В ЧАСОВОМУ РЯДУ ВІДСУТНЯ.")

# ============================================================
# 5. Розкласти динамічний ряд на складові (Декомпозиція)
# ============================================================
# Модель: Адитивна (для даних із приблизно постійною амплітудою сезонності)

decomposition = seasonal_decompose(ts, model='additive', period=12)

print("\n" + "="*70)
print("5. Декомпозиція динамічного ряду")
print("="*70)

decomposition.plot()
plt.suptitle('Декомпозиція Часового Ряду', y=1.02)
plt.show()

print("\n--- Висновки про складові ряду ---")
print("1. Тренд: Чітко виражений, зростаючий.")
print("2. Сезонність: Регулярні коливання з періодом 12 (річний цикл).")
print("3. Залишок (Residual): Випадкова, неструктурована компонента.")


# ============================================================
# 6. Побудувати автокореляційну функцію (ACF) випадкової компоненти
# ============================================================

residuals = decomposition.resid.dropna() 

# Повторний Ljung-Box Test для перевірки залишків (чи є вони білим шумом)
ljung_box_resid_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
ljung_box_p_value = ljung_box_resid_test['lb_pvalue'].iloc[0]

print("\n" + "="*70)
print("6. Автокореляційна функція (ACF) та оцінка коректності")
print("="*70)

# Побудова ACF залишків
plot_acf(residuals, lags=20, title='Автокореляційна функція Залишків (Residuals)')
plt.xlabel('Лаг')
plt.show()

print(f"P-значення Ljung-Box Test (Lag 10): {ljung_box_p_value:.4f}")

if ljung_box_p_value > alpha:
    print(f"✅ Коректність розкладення: Залишки є 'білим шумом' (немає значущої автокореляції). Розкладення коректне.")
else:
    print(f"❌ Коректність розкладення: Залишки містять значущу автокореляцію. Розкладення некоректне.")