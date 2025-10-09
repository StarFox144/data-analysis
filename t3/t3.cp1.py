# Імпортуємо необхідні бібліотеки
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# 1. Імітація даних користувачів (поведінковий аналіз)


# - кількість переглянутих сторінок
# - середній час на сайті (хвилини)
# - кількість кліків
# - кількість покупок

data = {
    'page_views': [5, 12, 25, 40, 8, 60, 15, 70, 30, 100],
    'time_on_site': [2, 5, 10, 15, 3, 20, 7, 25, 12, 30],
    'clicks': [10, 25, 40, 70, 15, 100, 35, 130, 60, 200],
    'purchases': [0, 1, 2, 5, 0, 8, 1, 10, 3, 12]
}

df = pd.DataFrame(data)


# 2. Попередня обробка даних
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. Кластеризація (K-Means)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['cluster'] = clusters


# 4. Оцінка якості кластеризації
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {silhouette_avg:.3f}")


# 5. Візуалізація результатів
plt.scatter(df['page_views'], df['time_on_site'], c=df['cluster'], cmap='viridis', s=100)
plt.xlabel('Кількість переглядів сторінок')
plt.ylabel('Час на сайті (хв)')
plt.title('Кластеризація користувачів за поведінковими ознаками')
plt.show()
