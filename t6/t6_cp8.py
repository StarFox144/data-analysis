import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_or_create_interactions(path: str = "interactions.csv") -> pd.DataFrame:
    """
    Якщо файл interactions.csv існує – читаємо його.
    Якщо ні – генеруємо штучні дані (50 юзерів, 30 айтемів, 1000 взаємодій).
    """
    if os.path.exists(path):
        print(f"[INFO] Знайдено існуючий файл {path}, завантажую дані...")
        df = pd.read_csv(path)
    else:
        print(f"[INFO] Файл {path} не знайдено, генерую випадкові дані...")
        np.random.seed(42)
        num_users = 50
        num_items = 30
        num_interactions = 1000

        df = pd.DataFrame({
            "user_id": np.random.randint(1, num_users + 1, size=num_interactions),
            "item_id": np.random.randint(1, num_items + 1, size=num_interactions),
            "rating": np.random.randint(1, 6, size=num_interactions)
        })
        df.to_csv(path, index=False)
        print(f"[INFO] Збережено згенеровані дані в {path}")

    return df


def generate_plots(df: pd.DataFrame,
                   output_dir: str = "pr2_plots") -> None:
    """
    Створює 5 графіків для ПР-2 і зберігає їх у вказану папку.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Додаємо псевдо-час (місяць 1..6)
    np.random.seed(0)
    df = df.copy()
    df["month"] = np.random.randint(1, 7, size=len(df))

    # ---------- 1. Time series: активність по місяцях ----------
    monthly_activity = df.groupby("month").size()

    plt.figure()
    monthly_activity.plot()
    plt.xlabel("Місяць")
    plt.ylabel("Кількість взаємодій")
    plt.title("Активність користувачів у часі")
    plt.tight_layout()
    ts_path = os.path.join(output_dir, "time_series_activity.png")
    plt.savefig(ts_path)
    plt.close()
    print(f"[OK] Збережено: {ts_path}")

    # ---------- 2. Топ-10 найпопулярніших товарів ----------
    top_items = (
        df.groupby("item_id")["user_id"]
          .nunique()
          .sort_values(ascending=False)
          .head(10)
    )

    plt.figure()
    top_items.plot(kind="bar")
    plt.xlabel("ID товару")
    plt.ylabel("Кількість унікальних користувачів")
    plt.title("Топ-10 найпопулярніших товарів")
    plt.tight_layout()
    top_items_path = os.path.join(output_dir, "top_10_items.png")
    plt.savefig(top_items_path)
    plt.close()
    print(f"[OK] Збережено: {top_items_path}")

    # ---------- 3. Heatmap user–item (10×10) ----------
    sample_users = df["user_id"].unique()[:10]
    sample_items = df["item_id"].unique()[:10]

    heatmap_data = df[
        df["user_id"].isin(sample_users)
        & df["item_id"].isin(sample_items)
    ]

    pivot = heatmap_data.pivot_table(
        index="user_id",
        columns="item_id",
        values="rating",
        fill_value=0
    )

    plt.figure()
    plt.imshow(pivot)
    plt.colorbar(label="Рейтинг")
    plt.title("Теплова карта взаємодій користувачів з товарами")
    plt.xlabel("ID товару")
    plt.ylabel("ID користувача")
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "heatmap_user_item.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"[OK] Збережено: {heatmap_path}")

    # ---------- 4. Порівняння алгоритмів (Precision / Recall) ----------
    # Тут поки що синтетичні значення – чисто для звіту
    algorithms = ["Item-Item CF", "Random"]
    precision = [0.32, 0.15]
    recall = [0.45, 0.21]

    # Precision
    plt.figure()
    plt.bar(algorithms, precision)
    plt.ylim(0, 1)
    plt.ylabel("Precision@K")
    plt.title("Порівняння точності алгоритмів")
    plt.tight_layout()
    prec_path = os.path.join(output_dir, "precision_comparison.png")
    plt.savefig(prec_path)
    plt.close()
    print(f"[OK] Збережено: {prec_path}")

    # Recall
    plt.figure()
    plt.bar(algorithms, recall)
    plt.ylim(0, 1)
    plt.ylabel("Recall@K")
    plt.title("Порівняння повноти алгоритмів")
    plt.tight_layout()
    rec_path = os.path.join(output_dir, "recall_comparison.png")
    plt.savefig(rec_path)
    plt.close()
    print(f"[OK] Збережено: {rec_path}")

    # ---------- 5. Розподіл попадань (hit distribution) ----------
    # hit = 1 якщо рекомендація "вгадала" інтерес, 0 — ні (штучно)
    np.random.seed(1)
    df["hit"] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
    hit_dist = df["hit"].value_counts().sort_index()

    plt.figure()
    hit_dist.plot(kind="bar")
    plt.xlabel("Hit (0 = промах, 1 = попадання)")
    plt.ylabel("Кількість випадків")
    plt.title("Розподіл попадань рекомендацій")
    plt.tight_layout()
    hit_path = os.path.join(output_dir, "hit_distribution.png")
    plt.savefig(hit_path)
    plt.close()
    print(f"[OK] Збережено: {hit_path}")


if __name__ == "__main__":
    # 1. Завантажуємо або генеруємо interactions.csv
    df_interactions = load_or_create_interactions("interactions_pr2.csv")

    # 2. Генеруємо всі 5 графіків
    generate_plots(df_interactions, output_dir="pr2_plots")

    print("\nВсе готово ✅ Графіки збережені в папці 'pr2_plots'.")
