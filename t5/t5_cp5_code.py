import pandas as pd
import numpy as np
from datetime import datetime

def generate_year_data(year: int, n_rows: int, exchange_rate: float) -> pd.DataFrame:
    """
    Генерує дані для одного року відповідно до варіанта 1:
    проекти зовнішньої реклами + розрахунок кварталу і вартостей.
    """
    rng = np.random.default_rng(seed=year)

    projects = ["Білборд центр", "LED-екран ТРЦ", "Реклама в метро",
                "Фасадний банер", "Брендування авто"]
    clients = ["АТБ", "Сільпо", "Епіцентр", "ROZETKA", "Comfy"]
    addresses = ["Київ, Хрещатик", "Львів, центр", "Харків, площа",
                 "Одеса, узбережжя", "Дніпро, проспект"]
    materials = ["Банер", "Плівка", "LED-модулі", "Каркас", "Фарба"]
    units = ["м²", "шт", "м"]

    # Усі дати року
    dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")

    # Базова таблиця
    df = pd.DataFrame({
        "Назва проекту": rng.choice(projects, n_rows),
        "Назва клієнта": rng.choice(clients, n_rows),
        "Адреса": rng.choice(addresses, n_rows),
        "Дата початку проекту": rng.choice(dates, n_rows),
        "Назва матеріалу": rng.choice(materials, n_rows),
        "Одиниця виміру": rng.choice(units, n_rows),
        "Ціна одиниці матеріалу в у.о.": rng.integers(10, 200, n_rows),
        "Кількість": rng.integers(1, 50, n_rows),
    })

    # Обчислювані поля
    df["Квартал"] = df["Дата початку проекту"].dt.quarter
    df["Ціна в грн"] = df["Ціна одиниці матеріалу в у.о."] * exchange_rate
    df["Загальна вартість матеріалу в у.о."] = (
        df["Ціна одиниці матеріалу в у.о."] * df["Кількість"]
    )
    df["Загальна вартість матеріалу в грн"] = df["Ціна в грн"] * df["Кількість"]
    df.insert(0, "Рік", year)

    return df


def main():
    # Курс валют (1 у.о. = 40 грн)
    exchange_rate = 40.0
    # Можна додати кілька років, якщо треба: [2023, 2024, 2025]
    years = [2024]

    excel_name = "t5_cp5.xlsx"

    with pd.ExcelWriter(
        excel_name,
        engine="xlsxwriter",
        datetime_format="yyyy-mm-dd"
    ) as writer:

        for year in years:
            df_year = generate_year_data(
                year=year,
                n_rows=120,          # скільки записів на рік згенерувати
                exchange_rate=exchange_rate
            )

            # Лист з "сирими" даними
            sheet_name = str(year)
            df_year.to_excel(writer, sheet_name=sheet_name, index=False, startrow=3)

            workbook = writer.book
            worksheet = writer.sheets[sheet_name]

            # Верхня частина аркуша: дата, курс, рік
            today = datetime.today().strftime("%d.%m.%Y")
            worksheet.write(0, 0, f"Сьогоднішня дата: {today}")
            worksheet.write(1, 0, f"Курс валюти (грн за 1 у.о.): {exchange_rate}")
            worksheet.write(2, 0, f"Рік: {year}")

            # Зведена таблиця: матеріал × клієнт × квартал (сума вартості в грн)
            pivot = pd.pivot_table(
                df_year,
                values="Загальна вартість матеріалу в грн",
                index="Назва матеріалу",                 # рядки – матеріали
                columns=["Назва клієнта", "Квартал"],    # стовпці – клієнт + квартал
                aggfunc="sum",
                fill_value=0
            )

            pivot_sheet_name = f"Зведена_{year}"
            pivot.to_excel(writer, sheet_name=pivot_sheet_name)

    print(f"Файл {excel_name} успішно створено.")


if __name__ == "__main__":
    main()
