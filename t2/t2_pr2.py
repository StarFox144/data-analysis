# === 1. Імпорт бібліотек ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# === 2. Завантаження даних ===
data = pd.read_csv("user_activity.csv", parse_dates=['date'])
data.set_index('date', inplace=True)
data = data.sort_index()

# === 3. Початковий графік ===
plt.figure(figsize=(10, 4))
plt.plot(data['views'], label='Активність користувачів')
plt.title('Початковий часовий ряд активності')
plt.xlabel('Дата')
plt.ylabel('Перегляди')
plt.legend()
plt.grid()
plt.show()

# === 4. Перевірка стаціонарності (ADF-тест) ===
result = adfuller(data['views'])
print("ADF-статистика:", result[0])
print("p-значення:", result[1])

if result[1] < 0.05:
    print("✅ Ряд стаціонарний")
    d = 0
else:
    print("❌ Ряд нестаціонарний — зробимо першу різницю")
    d = 1
    data['diff'] = data['views'].diff()
    plt.figure(figsize=(10, 4))
    plt.plot(data['diff'], color='orange', label='Після диференціювання')
    plt.legend()
    plt.title('Стаціонаризований ряд')
    plt.grid()
    plt.show()

# === 5. Підбір p та q вручну (через цикл) ===
best_aic = float("inf")
best_order = None
best_model = None

# обмежимо пошук 0..3 для p і q, щоб не було занадто довго
for p in range(4):
    for q in range(4):
        try:
            model = ARIMA(data['views'], order=(p, d, q))
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = (p, d, q)
                best_model = results
        except:
            continue

print(f"\n🔍 Найкраща модель ARIMA{best_order} з AIC = {best_aic:.2f}")

# === 6. Підсумок моделі ===
print(best_model.summary())

# === 7. Аналіз залишків ===
residuals = best_model.resid

plt.figure(figsize=(10, 4))
plt.plot(residuals, label='Залишки моделі')
plt.title('Залишки ARIMA')
plt.legend()
plt.grid()
plt.show()

sm.qqplot(residuals, line='s')
plt.title("Q-Q графік залишків")
plt.show()

sm.graphics.tsa.plot_acf(residuals, lags=20)
plt.title("Автокореляція залишків")
plt.show()

# === 8. Прогноз на 10 періодів ===
forecast_steps = 10
forecast = best_model.forecast(steps=forecast_steps)
forecast_index = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
forecast_series = pd.Series(forecast, index=forecast_index)

# === 9. Візуалізація прогнозу ===
plt.figure(figsize=(10, 4))
plt.plot(data['views'], label='Історичні дані')
plt.plot(forecast_series, label='Прогноз', color='red')
plt.title(f'Прогноз активності користувачів на {forecast_steps} днів уперед (ARIMA{best_order})')
plt.xlabel('Дата')
plt.ylabel('Перегляди')
plt.legend()
plt.grid()
plt.show()

print("\n📈 Прогноз на 10 днів уперед:")
print(forecast_series)


