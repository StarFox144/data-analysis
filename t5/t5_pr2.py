import pandas as pd
import numpy as np

# -----------------------------
# 1. Генерація факт-таблиці
# -----------------------------

rng = np.random.default_rng(4)
n_rows = 400

dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")

product_family = ["Candy", "Beverages", "Food"]
product_category = ["Candy", "Drink", "Food"]
product_subcategory = [
    "Chocolate candy", "Hard candy", "Milk", "Juice", "Fruit", "Snacks"
]

countries = ["USA", "Mexico"]
regions = ["West", "East", "North", "South"]
provinces = ["CA", "TX", "NY", "FL"]
cities = ["Los Angeles", "San Diego", "New York", "Miami"]

# Факт-таблиця продажів (sales)
df = pd.DataFrame({
    "Date": rng.choice(dates, n_rows),
    "Country": rng.choice(countries, n_rows),
    "Region": rng.choice(regions, n_rows),
    "Province": rng.choice(provinces, n_rows),
    "City": rng.choice(cities, n_rows),
    "ProductFamily": rng.choice(product_family, n_rows),
    "ProductCategory": rng.choice(product_category, n_rows),
    "ProductSubcategory": rng.choice(product_subcategory, n_rows),
    "UnitSales": rng.integers(1, 50, n_rows),
    "StoreCost": rng.uniform(1.0, 20.0, n_rows).round(2),
})

# -----------------------------
# 2. Вимір часу
# -----------------------------
df["Year"] = df["Date"].dt.year
df["Quarter"] = df["Date"].dt.quarter
df["Month"] = df["Date"].dt.month

# -----------------------------
# 3. Міра StoreSales
# -----------------------------
margin = rng.uniform(1.1, 1.8, n_rows).round(2)
df["StoreSales"] = (df["StoreCost"] * margin * df["UnitSales"]).round(2)

# -----------------------------
# 4. Вимір Product:
#    Candy → Chocolate candy
# -----------------------------
df_candy = df[
    (df["ProductCategory"] == "Candy") &
    (df["ProductSubcategory"] == "Chocolate candy")
].copy()

# -----------------------------
# 5. OLAP-куб (pivot)
# -----------------------------
pivot_candy = pd.pivot_table(
    df_candy,
    values=["UnitSales", "StoreSales", "StoreCost"],
    index=["Year", "Quarter"],   # Вимір Time
    columns=["Country"],         # Вимір Region (Country)
    aggfunc="sum",
    fill_value=0
)

# -----------------------------
# 6. Запис у Excel (XLSX)
# -----------------------------
excel_path = "olap_candy_chocolate_cube.xlsx"

with pd.ExcelWriter(excel_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
    df.to_excel(writer, sheet_name="Data", index=False)
    df_candy.to_excel(writer, sheet_name="Candy_Chocolate", index=False)
    pivot_candy.to_excel(writer, sheet_name="Cube_Candy_Chocolate")

print(f"✅ Файл створено: {excel_path}")
