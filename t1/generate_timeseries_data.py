# generate_timeseries_data.py
import pandas as pd
import numpy as np

# -------------------------------------------------------------
# 1. Параметри та генерація даних
# -------------------------------------------------------------
# Встановлюємо початкове значення для відтворюваності
np.random.seed(42) 
periods = 48  # 4 роки щомісячних даних
start_date = '2020-01-01'
csv_filename = 'user_behavior_data.csv'

# Створення індексу дат
dates = pd.date_range(start=start_date, periods=periods, freq='M')

# Генерація компонентів:
# 1. Тренд (зростаючий)
trend = np.arange(periods) * 1.5 + 50
# 2. Сезонність (місячні коливання)
seasonal = 10 * np.sin(np.linspace(0, 3 * np.pi, periods))
# 3. Шум (випадкова компонента)
noise = np.random.normal(0, 5, periods)

# Формування фінального ряду (Адитивна модель: Trend + Seasonal + Noise)
data_series = trend + seasonal + noise

# -------------------------------------------------------------
# 2. Створення DataFrame та Збереження
# -------------------------------------------------------------
df = pd.DataFrame({'Дата': dates, 'Показник_Активності': data_series})
df['Дата'] = df['Дата'].dt.strftime('%Y-%m-%d') # Форматування дати для CSV

df.to_csv(csv_filename, index=False)

print("="*70)
print(f"✅ Файл '{csv_filename}' успішно створено.")
print(f"   Розмір даних: {df.shape[0]} записів.")
print("="*70)