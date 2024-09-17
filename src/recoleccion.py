import os
import datetime
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


API_KEY = os.getenv("CALENDAR_API_KEY")

if not API_KEY:
    raise EnvironmentError("Set CALENDAR_API_KEY environment variable.")


# Funcion para obtener estacion del anio
def get_season(date):
    year = date.year
    if datetime.date(year, 12, 21) <= date <= datetime.date(year + 1, 3, 19):
        return "Invierno"
    elif datetime.date(year, 3, 20) <= date <= datetime.date(year, 6, 20):
        return "Primavera"
    elif datetime.date(year, 6, 21) <= date <= datetime.date(year, 9, 22):
        return "Verano"
    elif datetime.date(year, 9, 23) <= date <= datetime.date(year, 12, 20):
        return "Otoño"
    return "Sin estacion"


# Función para verificar si hay algún feriado durante la semana (de lunes a domingo)
def check_week_for_holiday(anio, semana, country="MX"):
    # Convertir la semana y el anio a datatime con fecha de lunes
    start_of_week = datetime.datetime.strptime(f"{anio} {semana} 1", "%G %V %u").date()
    # lista de fechas desde el lunes hasta el domingo
    week_dates = [start_of_week + datetime.timedelta(days=i) for i in range(7)]

    # Revisar cada día de la semana si es festivo
    for date in week_dates:
        url = "https://calendarific.com/api/v2/holidays"
        params = {
            "api_key": API_KEY,
            "country": country,
            "year": date.year,
            "month": date.month,
            "day": date.day,
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            holidays = response.json().get("response", {}).get("holidays", [])
            if holidays:
                return 1  # Al menos un dia festivo en la semana
    return 0  # sin festivos en la semana


# Función para calcular los límites de atípicos con base en cuartiles y etiquetar ventas especiales
def label_special_events(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    # Etiquetar con base en los valores superiores a los cuartiles
    df["EventoEspecial_" + column] = (df[column] > upper_bound).astype(int)
    return df


# Función para aniadir la columna de porcentaje de gasto en alimentos basado en el NSE
def add_food_spending_percentage(df, nse_data):
    # merge entre el dataframe original y el archivo de NSE con porcentaje de gasto en alimentos
    df = pd.merge(
        df, nse_data, how="left", left_on="Nivel Socioeconomico", right_on="NSE"
    )
    # Eliminar la columna duplicada 'NSE' después del merge
    df.drop(columns=["NSE"], inplace=True)
    return df


# Función para agregar las columnas de estaciones y feriados
def process_row(row):

    date = datetime.datetime.strptime(
        f"{int(row['Anio'])} {int(row['Semana'])} 1", "%G %V %u"
    ).date()

    # Obtener la estacion
    season = get_season(date)
    # obtener feriados durante la semana
    holiday = check_week_for_holiday(int(row["Anio"]), int(row["Semana"]))

    return pd.Series({"date": date, "season": season, "holiday": holiday})


# Main function to process the dataset
def main():

    df = pd.read_csv("data\\raw\\test_seriedatos_arquitectodatascience -.csv")

    # Cargar el archivo con los datos de porcentaje de gasto en alimentos por nivel socioeconómico
    nse_data = pd.read_csv("data\\raw\\gasto_alimentos.csv")

    # Reemplazar 0 con NaN en columnas relevantes
    # basado en las conclusiones de la exploracion de datos
    df[["Venta_piezas", "Venta_valor", "Precio"]] = df[
        ["Venta_piezas", "Venta_valor", "Precio"]
    ].replace(0, pd.NA)

    # Llenar los valores faltantes en 'Precio' y 'Venta_piezas'
    with pd.option_context("future.no_silent_downcasting", True):
        df["Precio"] = pd.to_numeric(
            df["Precio"].infer_objects().fillna(df["Precio"].mean())
        )
        df["Venta_piezas"] = pd.to_numeric(
            df["Venta_piezas"].infer_objects().fillna(df["Venta_piezas"].mean())
        )

    # Recalcular  'Venta_valor'
    df["Venta_valor"] = df["Venta_piezas"] * df["Precio"]
    df["Venta_valor"] = pd.to_numeric(df["Venta_valor"])

    # Aniadir las columnas de 'season' y 'holiday' basadas en 'Anio' y 'Semana'
    df[["date", "season", "holiday"]] = df.apply(process_row, axis=1)

    # Etiquetar los eventos especiales en 'Venta_piezas'
    df = label_special_events(df, "Venta_piezas")

    # Aniadir la columna de porcentaje de gasto en alimentos con base en el NSE
    df = add_food_spending_percentage(df, nse_data)

    # Guardar el dataset enriquecido
    df.to_csv("data\\dataset\\dataset.csv", index=False)

    print("Done.")


if __name__ == "__main__":
    main()
