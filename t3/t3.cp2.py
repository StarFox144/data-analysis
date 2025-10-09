
import sys
import subprocess

# Встановлення необхідних бібліотек
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandas", "numpy", "matplotlib", "statsmodels", "scipy"])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import pearsonr

print("✅ Усі бібліотеки імпортовано успішно!")

# 1. Дослідження структури часового ряду
np.random.seed(42)
days = pd.date_range('2025-01-01', periods=180)
visits = 500 + np.linspace(0, 100, 180) + 50 * np.sin(np.arange(180) * 2 * np.pi / 30) + np.random.normal(0, 10, 180)
data = pd.Series(visits, index=days)

decomposition = seasonal_decompose(data, model='additive', period=30)
decomposition.plot()
plt.suptitle('Дослідження структури часового ряду: тренд і сезонність', fontsize=12)
plt.show()

# 2. Побудова математичної моделі (ARIMA)
revenue = 10000 + np.linspace(0, 2000, 180) + 200 * np.sin(np.arange(180) * 2 * np.pi / 30) + np.random.normal(0, 100, 180)
revenue_series = pd.Series(revenue, index=days)

model = ARIMA(revenue_series, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

plt.figure(figsize=(8,4))
plt.plot(revenue_series, label='Фактичні дані')
plt.plot(model_fit.fittedvalues, label='Модель ARIMA', color='red')
plt.title('Побудова моделі економічного процесу (доходи користувачів)')
plt.legend()
plt.show()

# 3. Прогнозування майбутнього розвитку
forecast = model_fit.forecast(steps=30)
plt.figure(figsize=(8,4))
plt.plot(revenue_series, label='Історичні дані')
plt.plot(pd.date_range(days[-1], periods=30, freq='D'), forecast, label='Прогноз', color='orange')
plt.title('Прогноз доходу користувачів на майбутнє')
plt.legend()
plt.show()

# 4. Аналіз причинно-наслідкових зв’язків
ads = 100 + 10 * np.sin(np.arange(180) * 2 * np.pi / 15) + np.random.normal(0, 2, 180)
purchases = ads * 1.5 + np.random.normal(0, 5, 180)

corr, p_value = pearsonr(ads, purchases)
print(f'Кореляція між показами реклами та покупками: {corr:.3f}, p-value={p_value:.3f}')

plt.scatter(ads, purchases)
plt.xlabel('Кількість показів реклами')
plt.ylabel('Кількість покупок')
plt.title('Причинно-наслідковий зв’язок між процесами')
plt.show()

# 5. Згладжування та фільтрація
activity = 200 + 30 * np.sin(np.arange(180) * 2 * np.pi / 14) + np.random.normal(0, 8, 180)
activity_series = pd.Series(activity, index=days)

smoothed = activity_series.rolling(window=7).mean()

plt.figure(figsize=(8,4))
plt.plot(activity_series, label='Шумні дані')
plt.plot(smoothed, label='Згладжені дані (ковзне середнє)', color='red')
plt.title('Згладжування часового ряду активності користувачів')
plt.legend()
plt.show()
