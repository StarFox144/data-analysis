# recommender_system.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import matplotlib.pyplot as plt
import os

# ==============================
# 1. ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ
# ==============================

def load_data(path: str) -> pd.DataFrame:
    """
    Очікується CSV з колонками:
    user_id, item_id, rating (або event_strength)
    """
    df = pd.read_csv(path)
    # Перейменуємо колонку, якщо в тебе вона називається інакше
    if "rating" not in df.columns and "event_strength" in df.columns:
        df.rename(columns={"event_strength": "rating"}, inplace=True)

    # Залишимо тільки потрібні колонки
    df = df[["user_id", "item_id", "rating"]]

    # Приберемо пропуски
    df.dropna(inplace=True)

    return df


def encode_ids(df: pd.DataFrame):
    """
    Перетворюємо user_id, item_id у числові індекси
    (щоб працювати з матрицями/масивами).
    """
    user_ids = df["user_id"].unique()
    item_ids = df["item_id"].unique()

    user2idx = {u: i for i, u in enumerate(user_ids)}
    idx2user = {i: u for u, i in user2idx.items()}

    item2idx = {p: i for i, p in enumerate(item_ids)}
    idx2item = {i: p for p, i in item2idx.items()}

    df_enc = df.copy()
    df_enc["user_idx"] = df_enc["user_id"].map(user2idx)
    df_enc["item_idx"] = df_enc["item_id"].map(item2idx)

    return df_enc, user2idx, idx2user, item2idx, idx2item


def train_test_split_by_user(df_enc: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Розбиваємо дані за принципом:
    для кожного користувача частину взаємодій кидаємо в тест.
    """
    train_rows = []
    test_rows = []

    rng = np.random.RandomState(random_state)

    for user_idx, user_data in df_enc.groupby("user_idx"):
        if len(user_data) == 1:
            # Якщо у користувача одна взаємодія – залишимо її в train
            train_rows.append(user_data)
            continue

        user_train, user_test = train_test_split(
            user_data,
            test_size=test_size,
            random_state=rng.randint(0, 10_000)
        )
        train_rows.append(user_train)
        test_rows.append(user_test)

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)

    return train_df, test_df


# ==============================
# 2. ПОБУДОВА ITEM–ITEM МОДЕЛІ
# ==============================

def build_user_item_matrix(df_enc: pd.DataFrame):
    """
    Створюємо user–item матрицю (users x items),
    де значення – rating / event_strength.
    """
    n_users = df_enc["user_idx"].max() + 1
    n_items = df_enc["item_idx"].max() + 1

    user_item = np.zeros((n_users, n_items), dtype=np.float32)

    for _, row in df_enc.iterrows():
        u = int(row["user_idx"])
        i = int(row["item_idx"])
        r = float(row["rating"])
        user_item[u, i] = r

    return user_item


def build_item_similarity_matrix(user_item_matrix: np.ndarray):
    """
    Косинусна схожість між товарами (item–item).
    """
    item_item_sim = cosine_similarity(user_item_matrix.T)
    return item_item_sim


# ==============================
# 3. РЕКОМЕНДАЦІЇ КОРИСТУВАЧУ
# ==============================

def recommend_for_user_idx(
    user_idx: int,
    user_item_matrix,
    item_item_similarity,
    top_k: int = 10,
    filter_consumed: bool = True
):
    """
    Повертає (item_idx, score) для user_idx.
    """
    user_ratings = user_item_matrix[user_idx]
    scores = item_item_similarity.dot(user_ratings)

    if filter_consumed:
        scores[user_ratings > 0] = -np.inf

    top_items_idx = np.argsort(scores)[::-1][:top_k]
    recs = [(int(i), float(scores[i])) for i in top_items_idx if scores[i] != -np.inf]
    return recs


def recommend_for_user(
    user_id_raw,
    user2idx,
    idx2item,
    user_item_matrix,
    item_item_similarity,
    top_k: int = 10,
    already_consumed: bool = True
):
    """
    Обгортка: рекомендації за raw user_id.
    """
    if user_id_raw not in user2idx:
        raise ValueError(f"Користувач {user_id_raw} відсутній у даних")

    user_idx = user2idx[user_id_raw]
    recs_idx = recommend_for_user_idx(
        user_idx=user_idx,
        user_item_matrix=user_item_matrix,
        item_item_similarity=item_item_similarity,
        top_k=top_k,
        filter_consumed=already_consumed
    )

    recommendations = []
    for item_idx, score in recs_idx:
        item_id_raw = idx2item[item_idx]
        recommendations.append((item_id_raw, score))

    return recommendations


# ==============================
# 4. ОЦІНКА ЯКОСТІ (Precision@K, Recall@K)
# ==============================

def get_user_positive_items(df_enc: pd.DataFrame, user_col="user_idx", item_col="item_idx", rating_col="rating", threshold=0.0):
    """
    Повертає словник: user_idx -> множина item_idx, які вважаємо "позитивними".
    """
    user_pos_items = defaultdict(set)
    for _, row in df_enc.iterrows():
        if row[rating_col] > threshold:
            user_pos_items[int(row[user_col])].add(int(row[item_col]))
    return user_pos_items


def precision_recall_at_k(
    train_df, test_df, user_item_matrix, item_item_similarity, idx2user, idx2item, k: int = 10, threshold: float = 0.0
):
    """
    Обчислює середні Precision@K і Recall@K по всім користувачам.
    Повертає: mean_precision, mean_recall, per_user_metrics(DataFrame).
    """
    train_pos = get_user_positive_items(train_df, threshold=threshold)
    test_pos = get_user_positive_items(test_df, threshold=threshold)

    precisions = []
    recalls = []
    rows = []

    for user_idx in test_pos.keys():
        if user_idx not in train_pos or len(train_pos[user_idx]) == 0:
            continue

        user_id_raw = idx2user[user_idx]
        user_ratings = user_item_matrix[user_idx]
        scores = item_item_similarity.dot(user_ratings)
        scores[user_ratings > 0] = -np.inf

        top_items_idx = np.argsort(scores)[::-1][:k]
        rec_set = set(int(i) for i in top_items_idx if scores[i] != -np.inf)
        true_set = test_pos[user_idx]

        if not true_set:
            continue

        tp = len(rec_set & true_set)
        precision = tp / max(len(rec_set), 1)
        recall = tp / max(len(true_set), 1)

        precisions.append(precision)
        recalls.append(recall)

        rows.append({
            "user_idx": user_idx,
            "user_id": user_id_raw,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "true_positives": tp,
            "recommended": len(rec_set),
            "relevant_in_test": len(true_set)
        })

    mean_precision = float(np.mean(precisions)) if precisions else 0.0
    mean_recall = float(np.mean(recalls)) if recalls else 0.0

    per_user_df = pd.DataFrame(rows)
    return mean_precision, mean_recall, per_user_df


# ==============================
# 5. ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ У CSV
# ==============================

def save_recommendations_for_all_users(
    user_item_matrix,
    item_item_similarity,
    idx2user,
    idx2item,
    top_k: int = 10,
    output_path: str = "recommendations.csv"
):
    """
    Генерує рекомендації для ВСІХ користувачів і зберігає їх у CSV:
    user_id, item_id, rank, score
    """
    n_users = user_item_matrix.shape[0]
    rows = []

    for user_idx in range(n_users):
        user_id_raw = idx2user[user_idx]
        recs = recommend_for_user_idx(
            user_idx=user_idx,
            user_item_matrix=user_item_matrix,
            item_item_similarity=item_item_similarity,
            top_k=top_k,
            filter_consumed=True
        )
        for rank, (item_idx, score) in enumerate(recs, start=1):
            item_id_raw = idx2item[item_idx]
            rows.append({
                "user_idx": user_idx,
                "user_id": user_id_raw,
                "item_idx": item_idx,
                "item_id": item_id_raw,
                "rank": rank,
                "score": score
            })

    rec_df = pd.DataFrame(rows)
    rec_df.to_csv(output_path, index=False)
    print(f"✅ Рекомендації збережено у файл: {output_path}")


def save_metrics_summary(
    mean_precision: float,
    mean_recall: float,
    per_user_df: pd.DataFrame,
    summary_path: str = "metrics_summary.csv",
    per_user_path: str = "metrics_per_user.csv"
):
    """
    Зберігає загальні метрики та деталізовані по користувачу.
    """
    summary_df = pd.DataFrame([{
        "precision_at_k": mean_precision,
        "recall_at_k": mean_recall
    }])
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Загальні метрики збережено у файл: {summary_path}")

    per_user_df.to_csv(per_user_path, index=False)
    print(f"✅ Метрики по користувачах збережено у файл: {per_user_path}")


# ==============================
# 6. ПОБУДОВА ГРАФІКІВ
# ==============================

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_basic_charts(df: pd.DataFrame, output_dir: str = "plots"):
    """
    Створює базові графіки:
    - розподіл рейтингів
    - активність користувачів
    - популярність item-ів
    """
    ensure_dir(output_dir)

    # Розподіл рейтингів
    plt.figure()
    df["rating"].hist(bins=20)
    plt.title("Розподіл рейтингів")
    plt.xlabel("Рейтинг")
    plt.ylabel("Кількість")
    plt.tight_layout()
    ratings_path = os.path.join(output_dir, "ratings_hist.png")
    plt.savefig(ratings_path)
    plt.close()
    print(f"✅ Графік розподілу рейтингів збережено: {ratings_path}")

    # Кількість взаємодій на користувача
    plt.figure()
    df.groupby("user_id")["item_id"].count().hist(bins=20)
    plt.title("Кількість взаємодій на користувача")
    plt.xlabel("Кількість взаємодій")
    plt.ylabel("Кількість користувачів")
    plt.tight_layout()
    user_activity_path = os.path.join(output_dir, "user_activity_hist.png")
    plt.savefig(user_activity_path)
    plt.close()
    print(f"✅ Графік активності користувачів збережено: {user_activity_path}")

    # Популярність товарів
    plt.figure()
    df.groupby("item_id")["user_id"].count().hist(bins=20)
    plt.title("Популярність товарів (кількість користувачів на item)")
    plt.xlabel("Кількість користувачів")
    plt.ylabel("Кількість товарів")
    plt.tight_layout()
    item_pop_path = os.path.join(output_dir, "item_popularity_hist.png")
    plt.savefig(item_pop_path)
    plt.close()
    print(f"✅ Графік популярності товарів збережено: {item_pop_path}")


def plot_precision_recall_bar(precision: float, recall: float, output_dir: str = "plots"):
    """
    Простий стовпчиковий графік Precision@K та Recall@K.
    """
    ensure_dir(output_dir)

    plt.figure()
    metrics = ["Precision@K", "Recall@K"]
    values = [precision, recall]
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.title("Якість рекомендацій")
    plt.ylabel("Значення")
    plt.tight_layout()
    pr_path = os.path.join(output_dir, "precision_recall_bar.png")
    plt.savefig(pr_path)
    plt.close()
    print(f"✅ Графік Precision/Recall збережено: {pr_path}")


# ==============================
# 7. MAIN: ПРИКЛАД ВИКОРИСТАННЯ
# ==============================

if __name__ == "__main__":
    # 1. Завантаження
    data_path = "interactions.csv"  # <-- сюди клади свій файл
    df = load_data(data_path)
    print("Приклад даних:")
    print(df.head())

    # 2. Кодування ID
    df_enc, user2idx, idx2user, item2idx, idx2item = encode_ids(df)

    # 3. Train/Test split
    train_df, test_df = train_test_split_by_user(df_enc, test_size=0.2, random_state=42)
    print(f"Train interactions: {len(train_df)}, Test interactions: {len(test_df)}")

    # 4. Матриця користувач–товар (по train)
    user_item_train = build_user_item_matrix(train_df)

    # 5. Матриця схожості між товарами
    item_item_sim = build_item_similarity_matrix(user_item_train)

    # 6. Рекомендації для конкретного користувача (для демонстрації в консолі)
    example_user = df["user_id"].iloc[0]  # перший користувач з датасету
    recs = recommend_for_user(
        user_id_raw=example_user,
        user2idx=user2idx,
        idx2item=idx2item,
        user_item_matrix=user_item_train,
        item_item_similarity=item_item_sim,
        top_k=5,
    )

    print(f"\nРекомендації для користувача {example_user}:")
    for item_id, score in recs:
        print(f"  item = {item_id}, score = {score:.4f}")

    # 7. Оцінка якості (Precision@K, Recall@K)
    p_at_k, r_at_k, per_user_metrics = precision_recall_at_k(
        train_df=train_df,
        test_df=test_df,
        user_item_matrix=user_item_train,
        item_item_similarity=item_item_sim,
        idx2user=idx2user,
        idx2item=idx2item,
        k=10,
        threshold=0.0,
    )

    print(f"\nСередній Precision@10 = {p_at_k:.4f}")
    print(f"Середній Recall@10    = {r_at_k:.4f}")

    # 8. Збереження рекомендацій для всіх користувачів
    save_recommendations_for_all_users(
        user_item_matrix=user_item_train,
        item_item_similarity=item_item_sim,
        idx2user=idx2user,
        idx2item=idx2item,
        top_k=10,
        output_path="recommendations.csv"
    )

    # 9. Збереження метрик
    save_metrics_summary(
        mean_precision=p_at_k,
        mean_recall=r_at_k,
        per_user_df=per_user_metrics,
        summary_path="metrics_summary.csv",
        per_user_path="metrics_per_user.csv"
    )

    # 10. Побудова графіків
    plot_basic_charts(df, output_dir="plots")
    plot_precision_recall_bar(p_at_k, r_at_k, output_dir="plots")
