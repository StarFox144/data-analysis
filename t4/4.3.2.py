# --- Демонстрація прогнозу для Випадкового блукання з дрейфом ---
import numpy as np
import matplotlib.pyplot as plt
# Параметри моделі
delta = 0.1  # Дрейф (напрям)
sigma = 0.5  # Стандартне відхилення шуму
T = 50       # Довжина ряду для імітації
t_forecast = 40 # Момент часу, коли робиться прогноз
tau = 10     # Горизонт прогнозу

# Імітація ряду
np.random.seed(42)
epsilon = np.random.normal(0, sigma, T)
Y = np.zeros(T)
Y[0] = 5  # Початкове значення
for i in range(1, T):
    Y[i] = delta + Y[i-1] + epsilon[i]

Y_t = Y[t_forecast - 1] # Значення в момент t

# Прогноз (за формулою Y_t + tau * delta)
forecast_points = np.arange(1, tau + 1)
Y_forecast = Y_t + forecast_points * delta

# Побудова графіка
plt.figure(figsize=(10, 4))
plt.plot(range(1, T + 1), Y, label='Імітований часовий ряд ($Y_t$)', color='darkblue')
plt.plot(range(t_forecast, t_forecast + tau + 1), [Y_t] + list(Y_forecast), 
         linestyle='--', marker='o', color='red', label=f'Прогноз ($\hat{{y}}_{{t}}(\\tau)$)')
plt.axvline(x=t_forecast, color='gray', linestyle=':', label='Момент прогнозу ($t$)')
plt.title(f'Випадкове блукання з дрейфом (δ={delta}) та прогноз')
plt.xlabel('Час (t)')
plt.ylabel('Y')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
print("-" * 50)

# Розрахунок MSE для різних tau
print(f"Демонстрація Середньої Квадратичної Похибки (MSE):")
print(f"   σ² (Дисперсія шуму): {sigma**2:.4f}")
print(f"   Горизонт прогнозу (τ) | MSE (τ * σ²)")
print(f"   -----------------------|-----------------")
for t in [1, 5, 10]:
    mse = t * (sigma**2)
    print(f"   {t:20} | {mse:.4f}")