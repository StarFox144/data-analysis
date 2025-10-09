import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Крок 1: Визначення змінних та функцій належності (без змін) ---

# Вхідні змінні (Antecedents)
watch_time = ctrl.Antecedent(np.arange(0, 101, 1), 'watch_time')
rating = ctrl.Antecedent(np.arange(0, 11, 1), 'rating')
frequency = ctrl.Antecedent(np.arange(0, 31, 1), 'frequency')

# Вихідна змінна (Consequent)
recommendation_strength = ctrl.Consequent(np.arange(0, 101, 1), 'recommendation_strength')

# Функції належності
watch_time['low'] = fuzz.trimf(watch_time.universe, [0, 0, 40])
watch_time['medium'] = fuzz.trimf(watch_time.universe, [20, 50, 80])
watch_time['high'] = fuzz.trimf(watch_time.universe, [60, 100, 100])

rating['low'] = fuzz.trimf(rating.universe, [0, 0, 5])
rating['medium'] = fuzz.trimf(rating.universe, [3, 5, 7])
rating['high'] = fuzz.trimf(rating.universe, [6, 10, 10])

frequency['low'] = fuzz.trimf(frequency.universe, [0, 0, 7])
frequency['medium'] = fuzz.trimf(frequency.universe, [4, 10, 16])
frequency['high'] = fuzz.trimf(frequency.universe, [12, 30, 30])

# Функції належності для 'recommendation_strength'
recommendation_strength['weak'] = fuzz.trimf(recommendation_strength.universe, [0, 0, 40])
recommendation_strength['medium'] = fuzz.trimf(recommendation_strength.universe, [20, 50, 80])
recommendation_strength['strong'] = fuzz.trimf(recommendation_strength.universe, [60, 100, 100])

# --- Крок 2: Збереження графіків функцій належності (ВИПРАВЛЕНО) ---

print("Збереження графіків функцій належності...")

# Дозволяємо .view() створити графік, потім зберігаємо його
watch_time.view()
plt.gcf().savefig('mf_watch_time.png')
plt.close()

rating.view()
plt.gcf().savefig('mf_rating.png')
plt.close()

frequency.view()
plt.gcf().savefig('mf_frequency.png')
plt.close()

recommendation_strength.view()
plt.gcf().savefig('mf_recommendation_strength.png')
plt.close()
print("Графіки функцій належності збережено.")

# --- Крок 3: Визначення правил та створення системи (без змін) ---

rule1 = ctrl.Rule(rating['high'] | watch_time['high'], recommendation_strength['strong'])
rule2 = ctrl.Rule(rating['medium'] & watch_time['medium'] & frequency['high'], recommendation_strength['strong'])
rule3 = ctrl.Rule(rating['medium'] & watch_time['medium'] & frequency['medium'], recommendation_strength['medium'])
rule4 = ctrl.Rule(rating['low'] & watch_time['low'], recommendation_strength['weak'])
rule5 = ctrl.Rule(frequency['low'], recommendation_strength['weak'])

recommendation_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
recommendation_simulation = ctrl.ControlSystemSimulation(recommendation_ctrl)

# --- Крок 4: Симуляція та збереження результатів (ВИПРАВЛЕНО) ---

# Приклад 1: Активний користувач
print("\nРозрахунок для активного користувача...")
recommendation_simulation.input['watch_time'] = 95
recommendation_simulation.input['rating'] = 8
recommendation_simulation.input['frequency'] = 20
recommendation_simulation.compute()
print(f"Сила рекомендації: {recommendation_simulation.output.get('recommendation_strength', 0):.2f}%")

# Дозволяємо .view() створити графік, потім зберігаємо його
recommendation_strength.view(sim=recommendation_simulation)
plt.gcf().savefig('result_active_user.png')
plt.close()
print("Графік результату для активного користувача збережено.")


# Приклад 2: Пасивний користувач
print("\nРозрахунок для пасивного користувача...")
recommendation_simulation.input['watch_time'] = 20
recommendation_simulation.input['rating'] = 4
recommendation_simulation.input['frequency'] = 2
recommendation_simulation.compute()
print(f"Сила рекомендації: {recommendation_simulation.output.get('recommendation_strength', 0):.2f}%")

# Дозволяємо .view() створити графік, потім зберігаємо його
recommendation_strength.view(sim=recommendation_simulation)
plt.gcf().savefig('result_passive_user.png')
plt.close()
print("Графік результату для пасивного користувача збережено.")

# --- Крок 5: Створення та збереження 3D-поверхні відгуку (без змін) ---

print("\nСтворення 3D-поверхні відгуку...")
x_vals = np.linspace(rating.universe.min(), rating.universe.max(), 30)
y_vals = np.linspace(watch_time.universe.min(), watch_time.universe.max(), 30)
x, y = np.meshgrid(x_vals, y_vals)
z = np.zeros_like(x)

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        recommendation_simulation.input['rating'] = x[i, j]
        recommendation_simulation.input['watch_time'] = y[i, j]
        recommendation_simulation.input['frequency'] = 10
        try:
            recommendation_simulation.compute()
            z[i, j] = recommendation_simulation.output.get('recommendation_strength', 0)
        except Exception as e:
            z[i, j] = 0

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                       linewidth=0.4, antialiased=True)
ax.set_xlabel('Рейтинг (0-10)')
ax.set_ylabel('Час перегляду (%)')
ax.set_zlabel('Сила рекомендації (%)')
ax.set_title('Поверхня відгуку системи рекомендацій')
fig.colorbar(surf, shrink=0.5, aspect=5)
fig.savefig('surface_3d_plot.png')
plt.close(fig)

print("3D-графік поверхні відгуку збережено як 'surface_3d_plot.png'.")