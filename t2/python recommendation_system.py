import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    roc_auc_score, precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

# Налаштування візуалізації
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. ГЕНЕРАЦІЯ СИНТЕТИЧНИХ ДАНИХ ПРО ПОВЕДІНКУ КОРИСТУВАЧІВ
# ============================================================================

def generate_user_behavior_data(n_samples=5000, random_state=42):
    """
    Генерує синтетичні дані про поведінку користувачів
    
    Параметри:
    - n_samples: кількість записів
    - random_state: для відтворюваності результатів
    
    Повертає: DataFrame з даними про користувачів
    """
    np.random.seed(random_state)
    
    # Поведінкові характеристики
    time_on_site = np.random.exponential(scale=5, size=n_samples)  # хвилини
    pages_viewed = np.random.poisson(lam=3, size=n_samples)
    previous_purchases = np.random.poisson(lam=2, size=n_samples)
    days_since_last_visit = np.random.exponential(scale=7, size=n_samples)
    session_count = np.random.poisson(lam=5, size=n_samples)
    
    # Категоріальні дані
    device_type = np.random.choice(['mobile', 'desktop', 'tablet'], 
                                   size=n_samples, p=[0.5, 0.35, 0.15])
    time_of_day = np.random.choice(['morning', 'afternoon', 'evening', 'night'],
                                   size=n_samples, p=[0.2, 0.3, 0.35, 0.15])
    user_segment = np.random.choice(['new', 'regular', 'vip'],
                                    size=n_samples, p=[0.4, 0.45, 0.15])
    
    # Створення цільової змінної (клік на рекомендацію)
    # Логіка: більше часу на сайті, більше переглядів, попередні покупки збільшують ймовірність кліку
    click_probability = (
        0.1 +  # базова ймовірність
        0.05 * np.minimum(time_on_site / 10, 1) +  # час на сайті
        0.04 * np.minimum(pages_viewed / 5, 1) +   # перегляди
        0.06 * np.minimum(previous_purchases / 3, 1) +  # покупки
        0.03 * (device_type == 'desktop') +  # десктоп користувачі активніші
        0.05 * (user_segment == 'vip') +  # VIP активніші
        0.02 * (time_of_day == 'evening')  # вечір - найактивніший час
    )
    
    # Додаємо шум
    click_probability += np.random.normal(0, 0.1, n_samples)
    click_probability = np.clip(click_probability, 0, 1)
    
    # Генеруємо бінарну мітку
    clicked = (np.random.random(n_samples) < click_probability).astype(int)
    
    # Створюємо DataFrame
    data = pd.DataFrame({
        'time_on_site': time_on_site,
        'pages_viewed': pages_viewed,
        'previous_purchases': previous_purchases,
        'days_since_last_visit': days_since_last_visit,
        'session_count': session_count,
        'device_type': device_type,
        'time_of_day': time_of_day,
        'user_segment': user_segment,
        'clicked': clicked
    })
    
    return data

# ============================================================================
# 2. ПОПЕРЕДНЯ ОБРОБКА ДАНИХ
# ============================================================================

def preprocess_data(data):
    """
    Підготовка даних: one-hot encoding для категоріальних змінних
    """
    # Створюємо копію
    df = data.copy()
    
    # One-hot encoding для категоріальних змінних
    df = pd.get_dummies(df, columns=['device_type', 'time_of_day', 'user_segment'], 
                        drop_first=True)
    
    return df

# ============================================================================
# 3. ПОБУДОВА МОДЕЛІ ЛОГІСТИЧНОЇ РЕГРЕСІЇ
# ============================================================================

def build_logistic_regression_model(X_train, y_train, X_test, y_test):
    """
    Будує та навчає модель логістичної регресії
    """
    # Стандартизація ознак
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Створення та навчання моделі
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Прогнозування
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    return model, scaler, y_pred, y_pred_proba, X_train_scaled, X_test_scaled

# ============================================================================
# 4. ОЦІНКА МОДЕЛІ
# ============================================================================

def evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, 
                   y_pred, y_pred_proba, feature_names):
    """
    Комплексна оцінка моделі з візуалізацією
    """
    print("="*80)
    print("ОЦІНКА ПРОГНОСТИЧНОЇ МОДЕЛІ")
    print("="*80)
    
    # 1. Базові метрики
    print("\n1. КЛАСИФІКАЦІЙНИЙ ЗВІТ:")
    print("-"*80)
    print(classification_report(y_test, y_pred, 
                               target_names=['Не клікнув', 'Клікнув']))
    
    # 2. ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n2. ROC-AUC SCORE: {roc_auc:.4f}")
    
    # 3. Крос-валідація
    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                cv=5, scoring='roc_auc')
    print(f"\n3. КРОС-ВАЛІДАЦІЯ (5-fold):")
    print(f"   Середній ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # 4. Важливість ознак
    print(f"\n4. ВАЖЛИВІСТЬ ОЗНАК (коефіцієнти моделі):")
    print("-"*80)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', ascending=False)
    print(feature_importance.to_string(index=False))
    
    # 5. Візуалізація
    create_evaluation_plots(y_test, y_pred, y_pred_proba, feature_importance)
    
    return roc_auc, feature_importance

def create_evaluation_plots(y_test, y_pred, y_pred_proba, feature_importance):
    """
    Створює візуалізації для оцінки моделі
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Матриця помилок (Confusion Matrix)
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Матриця помилок', fontsize=14, fontweight='bold')
    plt.ylabel('Справжнє значення')
    plt.xlabel('Прогнозоване значення')
    
    # 2. ROC-крива
    ax2 = plt.subplot(2, 3, 2)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC крива (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Випадкова модель')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-крива', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    # 3. Precision-Recall крива
    ax3 = plt.subplot(2, 3, 3)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR крива (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall крива', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    
    # 4. Розподіл ймовірностей
    ax4 = plt.subplot(2, 3, 4)
    plt.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.6, label='Не клікнув', color='blue')
    plt.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.6, label='Клікнув', color='red')
    plt.xlabel('Прогнозована ймовірність')
    plt.ylabel('Частота')
    plt.title('Розподіл прогнозованих ймовірностей', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 5. Важливість ознак (топ-10)
    ax5 = plt.subplot(2, 3, 5)
    top_features = feature_importance.head(10)
    colors = ['green' if x > 0 else 'red' for x in top_features['coefficient']]
    plt.barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Коефіцієнт')
    plt.title('Топ-10 найважливіших ознак', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(alpha=0.3, axis='x')
    
    # 6. Calibration plot
    ax6 = plt.subplot(2, 3, 6)
    fraction_of_positives, mean_predicted_value = \
        calibration_curve_custom(y_test, y_pred_proba, n_bins=10)
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label='Логістична регресія')
    plt.plot([0, 1], [0, 1], "k--", label='Ідеальна калібрація')
    plt.xlabel('Середня прогнозована ймовірність')
    plt.ylabel('Частка позитивних класів')
    plt.title('Калібраційна крива', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    print("\n" + "="*80)
    print("Графіки збережено у файл: model_evaluation.png")
    print("="*80)
    plt.show()

def calibration_curve_custom(y_true, y_prob, n_bins=10):
    """Спрощена версія calibration curve"""
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    bin_sums = np.bincount(binids, weights=y_prob, minlength=n_bins)
    bin_true = np.bincount(binids, weights=y_true, minlength=n_bins)
    bin_total = np.bincount(binids, minlength=n_bins)
    
    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    
    return prob_true, prob_pred

# ============================================================================
# 5. ІНТЕРПРЕТАЦІЯ РЕЗУЛЬТАТІВ
# ============================================================================

def interpret_results(feature_importance):
    """
    Інтерпретація коефіцієнтів моделі
    """
    print("\n" + "="*80)
    print("ІНТЕРПРЕТАЦІЯ РЕЗУЛЬТАТІВ")
    print("="*80)
    
    print("\nПОЗИТИВНІ ФАКТОРИ (збільшують ймовірність кліку):")
    print("-"*80)
    positive = feature_importance[feature_importance['coefficient'] > 0].head(5)
    for idx, row in positive.iterrows():
        print(f"  • {row['feature']:<30} → +{row['coefficient']:.4f}")
    
    print("\nНЕГАТИВНІ ФАКТОРИ (зменшують ймовірність кліку):")
    print("-"*80)
    negative = feature_importance[feature_importance['coefficient'] < 0].tail(5)
    for idx, row in negative.iterrows():
        print(f"  • {row['feature']:<30} → {row['coefficient']:.4f}")
    
    print("\n" + "="*80)

# ============================================================================
# 6. ПРИКЛАД ВИКОРИСТАННЯ МОДЕЛІ
# ============================================================================

def predict_for_new_user(model, scaler, feature_names):
    """
    Приклад прогнозування для нового користувача
    """
    print("\n" + "="*80)
    print("ПРИКЛАД ПРОГНОЗУВАННЯ ДЛЯ НОВОГО КОРИСТУВАЧА")
    print("="*80)
    
    # Створюємо приклад нового користувача
    new_user = pd.DataFrame({
        'time_on_site': [8.5],
        'pages_viewed': [5],
        'previous_purchases': [3],
        'days_since_last_visit': [2.0],
        'session_count': [7],
        'device_type_mobile': [1],
        'device_type_tablet': [0],
        'time_of_day_evening': [1],
        'time_of_day_morning': [0],
        'time_of_day_night': [0],
        'user_segment_regular': [0],
        'user_segment_vip': [1]
    })
    
    # Масштабування
    new_user_scaled = scaler.transform(new_user)
    
    # Прогноз
    prediction = model.predict(new_user_scaled)[0]
    probability = model.predict_proba(new_user_scaled)[0, 1]
    
    print("\nХарактеристики користувача:")
    print(f"  • Час на сайті: 8.5 хвилин")
    print(f"  • Переглянуто сторінок: 5")
    print(f"  • Попередні покупки: 3")
    print(f"  • Днів з останнього візиту: 2")
    print(f"  • Кількість сесій: 7")
    print(f"  • Пристрій: Mobile")
    print(f"  • Час доби: Evening")
    print(f"  • Сегмент: VIP")
    
    print(f"\nРЕЗУЛЬТАТ ПРОГНОЗУ:")
    print(f"  • Ймовірність кліку: {probability:.2%}")
    print(f"  • Прогноз: {'КЛІКНЕ ✓' if prediction == 1 else 'НЕ КЛІКНЕ ✗'}")
    print("="*80)

# ============================================================================
# ГОЛОВНА ФУНКЦІЯ
# ============================================================================

def main():
    print("\n" + "="*80)
    print("СИСТЕМА РЕКОМЕНДАЦІЙ НА ОСНОВІ ПОВЕДІНКОВОГО АНАЛІЗУ")
    print("Використання логістичної регресії")
    print("="*80 + "\n")
    
    # 1. Генерація даних
    print("Крок 1: Генерація синтетичних даних...")
    data = generate_user_behavior_data(n_samples=5000)
    print(f"✓ Згенеровано {len(data)} записів")
    print(f"✓ Розподіл класів: {data['clicked'].value_counts().to_dict()}")
    
    # 2. Попередня обробка
    print("\nКрок 2: Попередня обробка даних...")
    data_processed = preprocess_data(data)
    print(f"✓ Кількість ознак після обробки: {data_processed.shape[1] - 1}")
    
    # 3. Розділення на train/test
    print("\nКрок 3: Розділення на навчальну та тестову вибірки...")
    X = data_processed.drop('clicked', axis=1)
    y = data_processed['clicked']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Навчальна вибірка: {len(X_train)} записів")
    print(f"✓ Тестова вибірка: {len(X_test)} записів")
    
    # 4. Побудова моделі
    print("\nКрок 4: Побудова та навчання моделі логістичної регресії...")
    model, scaler, y_pred, y_pred_proba, X_train_scaled, X_test_scaled = \
        build_logistic_regression_model(X_train, y_train, X_test, y_test)
    print("✓ Модель успішно навчена")
    
    # 5. Оцінка моделі
    print("\nКрок 5: Оцінка моделі...")
    roc_auc, feature_importance = evaluate_model(
        model, X_train_scaled, y_train, X_test_scaled, y_test,
        y_pred, y_pred_proba, X.columns.tolist()
    )
    
    # 6. Інтерпретація
    interpret_results(feature_importance)
    
    # 7. Приклад використання
    predict_for_new_user(model, scaler, X.columns.tolist())
    
    print("\n" + "="*80)
    print("ПРОЄКТ ЗАВЕРШЕНО УСПІШНО!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()