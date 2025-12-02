import pandas as pd
import matplotlib.pyplot as plt

# 1. Завантаження даних
df = pd.read_csv("variant1_orders.csv", parse_dates=["order_date"])
print("Перші 5 записів:")
print(df.head())

# 2. Побудова OLAP-зрізу (ціна та кількість)
pivot = pd.pivot_table(
    df,
    values=["price", "quantity"],
    index=["order_date"],
    columns=["category"],
    aggfunc="sum"
)

print("\nOLAP-зріз:")
print(pivot)

# 3. Збереження зведеної таблиці у CSV
pivot.to_csv("olap_pivot_variant1.csv")

# 4. Побудова діаграми (сума продажів по датах)
pivot["price"].plot(title="Сума продажів по датах")
plt.xlabel("Дата")
plt.ylabel("Сума цін")
plt.grid(True)
plt.tight_layout()
plt.savefig("olap_chart_variant1.png")
plt.show()

print("Готово! Створено:")
print(" - olap_pivot_variant1.csv")
print(" - olap_chart_variant1.png")
