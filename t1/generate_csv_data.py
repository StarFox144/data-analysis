import pandas as pd
import numpy as np

# ============================================================
# ГЕНЕРАЦІЯ РЕАЛІСТИЧНИХ ДАНИХ КОРИСТУВАЧІВ
# ============================================================

print("="*70)
print("ГЕНЕРАЦІЯ ДАНИХ ДЛЯ ПОВЕДІНКОВОГО АНАЛІЗУ")
print("="*70)

# Встановлюємо seed для відтворюваності
np.random.seed(42)

# Кількість користувачів (30 - оптимально для регресії)
n_users = 30

# --- Генерація незалежної змінної ---
# Час на сайті: від 5 до 60 хвилин
time_on_site = np.random.uniform(5, 60, n_users)

# --- Генерація залежної змінної з реалістичним зв'язком ---
# Формула: Покупки = 0.5 + 0.15 × Час + випадковий_шум
# Це створює помітний позитивний зв'язок
noise = np.random.normal(0, 0.8, n_users)
purchases = 0.5 + 0.15 * time_on_site + noise

# Округлення та видалення від'ємних значень
purchases = np.maximum(0, np.round(purchases))

# --- Створення датасету з додатковими метриками ---
data = {
    'ID_користувача': range(1, n_users + 1),
    'Час_на_сайті_хв': np.round(time_on_site, 1),
    'Кількість_покупок': purchases.astype(int),
    'Кількість_кліків': np.random.randint(10, 150, n_users),
    'Переглянуто_сторінок': np.random.randint(3, 50, n_users),
    'Додано_в_кошик': np.random.randint(0, 8, n_users)
}

df = pd.DataFrame(data)

# --- Збереження у CSV ---
csv_filename = 'user_behavior_data.csv'
df.to_csv(csv_filename, index=False, encoding='utf-8-sig')

print(f"\n✅ Файл успішно створено: {csv_filename}")
print(f"📊 Кількість користувачів: {n_users}")
print(f"📁 Розмір датасету: {df.shape[0]} рядків × {df.shape[1]} стовпців")

print("\n--- Перші 10 записів ---")
print(df.head(10))

print("\n--- Статистика основних змінних ---")
print(df[['Час_на_сайті_хв', 'Кількість_покупок']].describe())

print("\n" + "="*70)
print("✅ ГЕНЕРАЦІЯ ЗАВЕРШЕНА")
print(f"Файл '{csv_filename}' готовий до використання!")
print("="*70)   