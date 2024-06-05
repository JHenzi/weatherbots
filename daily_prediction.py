import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import requests
import pathlib
import json
from datetime import datetime
from meteostat import Stations, Daily
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pprint
import warnings
import uuid
import kalshi_python
from kalshi_python.models import *

warnings.filterwarnings("ignore")

# Load API keys
from dotenv import load_dotenv
import os

load_dotenv()
# print((os.getenv("VISUAL_CROSSING_API_KEY")))

# Define some constants
latitude = [40.79736, 41.78701, 30.1444, 25.7738]
longitude = [-73.97785, -87.77166, -97.66876, -80.1936]
cities = ["ny", "il", "tx", "fl"]
stations_ncei = ["USW00094728", "USW00014819", "USW00013904", "USC00086315"]
start_date = "2016-01-01"
# end_date = "2024-03-24"
time_steps = 60

# ## API Calls to collect historical weather data

# ### Open Meteo


def getDataFromOpenMeteo(latitude, longitude, startDate, endDate, fileName):
    # Data Source 1
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": startDate,
        "end_date": endDate,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "sunshine_duration",
            "precipitation_hours",
            "wind_speed_10m_max",
        ],
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    # print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    # print(f"Elevation {response.Elevation()} m asl")
    # print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    # print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
    daily_sunshine_duration = daily.Variables(2).ValuesAsNumpy()
    daily_precipitation_hours = daily.Variables(3).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(4).ValuesAsNumpy()

    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        )
    }
    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["temperature_2m_min"] = daily_temperature_2m_min
    daily_data["sunshine_duration"] = daily_sunshine_duration
    daily_data["precipitation_hours"] = daily_precipitation_hours
    daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max

    daily_dataframe = pd.DataFrame(data=daily_data)
    daily_dataframe.to_csv(
        "./Data/openMeteo_" + "_".join([fileName, startDate, "to", endDate]) + ".csv",
        index=False,
    )
    return daily_dataframe
    # print(daily_dataframe)


# ### Visual Crossing


def getDataFromVisualCrossing(latitude, longitude, startDate, endDate, fileName):
    url = (
        "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        + str(latitude)
        + "%2C"
        + str(longitude)
        + "/"
        + startDate
        + "/"
        + endDate
        + "?unitGroup=us&include=days&key="
        + os.getenv("VISUAL_CROSSING_API_KEY")
        + "&contentType=json"
    )
    # print(url)
    # print(
    #     "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/40.79736%2C-73.97785/2016-01-01/today?unitGroup=us&include=days&key=MHFU2QHX7NTY5RTWZPAT7VBXS&contentType=json"
    # )
    # "https://weather.visualcrossing.com/VisualCrosingWebServices/rest/services/timeline/
    # 40.79736%2C-73.97785/2016-01-01/today?unitGroup=us&include=days&key=MHFU2QHX7NTY5RTWZPAT7VBXS&contentType=json"

    payload = {}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)
    pathlib.Path(
        "./Data/visualCrossing_"
        + "_".join([fileName, startDate, "to", endDate])
        + ".json"
    ).write_bytes(response.content)

    print(response.text)


# ### Meteostat

# Get weather stations
stations = Stations()
required_stations = []
for i in range(len(latitude)):
    stations = stations.nearby(latitude[i], longitude[i])
    station = stations.fetch(1)
    required_stations.append(station)

# print(required_stations)


def getDataFromMeteostat(latitude, longitude, startDate, endDate, fileName):
    start = datetime.strptime(startDate, "%Y-%m-%d")
    end = datetime.strptime(endDate, "%Y-%m-%d")
    stations = Stations()
    stations = stations.nearby(latitude, longitude)
    station = stations.fetch(1)
    # Get daily data
    data = Daily(station, start, end)
    data = data.fetch()
    # print(data['time'])
    data.index.names = ["date"]
    data = data.add_suffix("_ms")
    data.to_csv(
        "./Data/meteoStat_" + "_".join([fileName, startDate, "to", endDate]) + ".csv"
    )
    return data
    # data.plot(y=['tavg', 'tmin', 'tmax'])
    # plt.show()


# ### NCEI


# Already downloaded manually as CSV
def getDataFromNCEI(station, startDate, endDate, fileName):
    base_url = "https://www.ncei.noaa.gov/access/services/data/v1?"
    # Define the query parameters
    params = {
        "dataset": "daily-summaries",
        "stations": station,
        "startDate": start_date,
        "endDate": end_date,
        "format": "json",
        "units": "standard",
    }

    response = requests.get(base_url, params=params)
    data = {}
    if response.status_code == 200:
        data = response.json()
    else:
        print("Error. Status code:", response.status_code)

    try:
        # Convert the json data from the NOAA to a dataframe
        data_df = pd.DataFrame.from_records(data)
        data_df.to_csv(
            "./Data/ncei_" + "_".join([fileName, startDate, "to", endDate]) + ".csv"
        )
    except:
        print("Couldn't get NCEI data for ", start_date, " to ", end_date, data)
        return


# ## Reading Data from the CSV and JSON Files created from the API calls


def readStoredJSONData(fileName):
    with open(fileName, "r") as file:
        # Reading from json file
        data = json.load(file)
    return data


def readStoredCSVData(fileName):
    df = pd.read_csv(fileName)
    return df

    # Store the merged_df DataFrame to disk for later computations
    # for i in range(len(cities)):
    # merged_dfs[i].to_pickle("./Data/merged_df_" + cities[i] + ".pkl")


def getDailyData(start_date, end_date):
    # Unpickle the DataFrames
    print(f"Getting daily data for end_date {end_date}\n")
    city_history_dfs = []

    for i in range(len(cities)):
        city_history_dfs.append(
            pd.read_pickle("./Data/merged_df_" + cities[i] + ".pkl")
        )

    print("Loaded DFs for " + str(len(city_history_dfs)) + " cities.\n")
    # print(city_history_dfs[0].info())
    print(city_history_dfs[0].tail())

    # # Get daily data and append it to existing Data Frame

    daily_data = []
    for i in range(len(latitude)):
        daily_data.append(
            getDataFromOpenMeteo(
                latitude[i], longitude[i], start_date, end_date, cities[i]
            )
        )

    visual_crossing_data = []
    for i in range(len(latitude)):
        visual_crossing_data.append(
            getDataFromVisualCrossing(
                latitude[i], longitude[i], start_date, end_date, cities[i]
            )
        )

        daily_ms_data = []
    for i in range(len(latitude)):
        daily_ms_data.append(
            getDataFromMeteostat(
                latitude[i], longitude[i], start_date, end_date, cities[i]
            )
        )

    for i in range(len(stations_ncei)):
        getDataFromNCEI(stations_ncei[i], start_date, end_date, cities[i])

    # Read all visual crossing files
    vc_data = []
    for i in range(len(latitude)):
        fileName = (
            "./Data/visualCrossing_"
            + "_".join([cities[i], start_date, "to", end_date])
            + ".json"
        )
        vc_data.append(readStoredJSONData(fileName))

    vc_dfs = []
    for cityData in vc_data:
        city_df = pd.DataFrame(cityData["days"])
        city_df = city_df[["datetime", "tempmax", "tempmin", "humidity", "windspeed"]]
        # print(city_df.info())
        vc_dfs.append(city_df)

    # Read all open meteo files
    om_dfs = []
    for i in range(len(latitude)):
        fileName = (
            "./Data/openMeteo_"
            + "_".join([cities[i], start_date, "to", end_date])
            + ".csv"
        )
        om_df = readStoredCSVData(fileName)
        # print(type(om_df['date'][0]))
        # om_df['date'] = om_df['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
        om_df["date"] = om_df["date"].apply(lambda x: x[:10])
        om_df.set_index("date")
        om_dfs.append(om_df)

    ms_dfs = []
    for i in range(len(latitude)):
        fileName = (
            "./Data/meteoStat_"
            + "_".join([cities[i], start_date, "to", end_date])
            + ".csv"
        )
        ms_df = readStoredCSVData(fileName)
        ms_df["date"] = ms_df["date"].apply(lambda x: x[:10])
        ms_df.set_index("date")
        ms_dfs.append(ms_df)

    ncei_dfs = []
    for i in range(len(cities)):
        fileName = (
            "./Data/ncei_" + "_".join([cities[i], start_date, "to", end_date]) + ".csv"
        )
        # print(fileName)
        ncei_df = readStoredCSVData(fileName)
        ncei_df.columns = map(str.lower, ncei_df.columns)
        # ncei_df['date'] = ncei_df['date'].apply(lambda x: x[:10])
        ncei_df = ncei_df.add_suffix("_ncei")
        ncei_df = ncei_df.rename(columns={"date_ncei": "date"})
        if ncei_df.size == 0:
            break
        ncei_df.set_index("date")
        ncei_dfs.append(ncei_df)
        # print(ncei_df.info())

    # Merge the daily data
    for vc_df in vc_dfs:
        vc_df.columns = ["date", "tmax_vc", "tmin_vc", "humi_vc", "wind_vc"]
        vc_df.set_index("date")

    for om_df in om_dfs:
        om_df.columns = ["date", "tmax_om", "tmin_om", "sund_om", "prec_om", "wind_om"]
        om_df.set_index("date")

    for ms_df in ms_dfs:
        ms_df.set_index("date")

    for ncei_df in ncei_dfs:
        if ncei_df.size == 0:
            break
        ncei_df.set_index("date")

    daily_merged_dfs = []

    for i in range(len(vc_dfs)):
        merged_df = pd.merge(vc_dfs[i], om_dfs[i], how="left", on="date")
        merged_df = pd.merge(merged_df, ms_dfs[i], how="left", on="date")
        # merged_df = pd.merge(merged_df, ncei_dfs[i], how='left', on='date')
        # , left_index=True, right_index=True)
        daily_merged_dfs.append(merged_df)

    # print(merged_dfs[0].info())
    # print(merged_dfs[0].tail())

    daily_merged_dfs[0].tail()

    city_history_dfs = []

    def appendDailyData():
        updatedCitiesDfs = []
        for i in range(len(cities)):
            city_history_dfs.append(
                pd.read_pickle("./Data/merged_df_" + cities[i] + ".pkl")
            )

        for i in range(len(cities)):
            updatedCitiesDfs.append(
                pd.concat([city_history_dfs[i], daily_merged_dfs[i]])
            )
            # print(updateCitiesDfs[i].tail())

        return updatedCitiesDfs

    latest_data = appendDailyData()
    print("Daily data has been downloaded!")

    # Store the merged_df DataFrame to disk for later computations
    for i in range(len(cities)):
        latest_data[i].to_pickle("./Data/prediction_merged_df_" + cities[i] + ".pkl")

    # Clean The Data
    def cleanAllData(
        fileNamePrefix="./Data/merged_df_", outputFilePrefix="./Data/data_cleaned_"
    ):
        # Unpickle the DataFrames
        city_history_dfs = []

        for i in range(len(cities)):
            city_history_dfs.append(pd.read_pickle(fileNamePrefix + cities[i] + ".pkl"))

        print("Loaded DFs for " + str(len(city_history_dfs)) + " cities.\n")
        # print(city_history_dfs[0].info())
        # print(city_history_dfs[0].tail())

        for i in range(len(cities)):
            # Figure out what all cols to keep
            allCols = city_history_dfs[i].columns.tolist()
            # print(len(allCols))
            # print(*allCols, sep="\n")
            # Keep the columns we need
            # date, tmax from all sources,
            # humidity from visual crossing,
            # precipitation from open meteo,
            # snow from meteo stats
            # sunny time from ncei
            ny_data = city_history_dfs[i][
                [
                    "date",
                    "tmax_vc",
                    "tmax_om",
                    "tmax_ms",
                    "tmax_ncei",
                    "humi_vc",
                    "prec_om",
                    "tmin_ms",
                    "tmin_ncei",
                ]
            ]
            # ny_data = city_history_dfs[i][['date', 'tmax_vc', 'tmax_om', 'tmax_ms', 'humi_vc', 'prec_om', 'tmin_ms', ]]
            ny_data.info()

            def cleanAndPreprocessData(df):
                columns_to_convert_to_farhenheit = ["tmax_om", "tmax_ms", "tmin_ms"]
                for column in columns_to_convert_to_farhenheit:
                    df[column] = df[column] * 9 / 5 + 32
                print(df.head())
                df["date"] = pd.to_datetime(df["date"])
                df["day"] = df["date"].dt.dayofyear
                df["tmax_avg"] = df[["tmax_vc", "tmax_om", "tmax_ms"]].mean(axis=1)
                df["tmin_avg"] = df[["tmin_ms"]].mean(axis=1)
                return df

            ny_data = cleanAndPreprocessData(ny_data)
            ny_data.to_pickle(outputFilePrefix + cities[i] + ".pkl")

    # cleanAllData()
    cleanAllData("./Data/prediction_merged_df_", "./Data/prediction_data_cleaned_")
    print("New data has been cleaned!")


def getPrediction(city, name_prefix="", offset=0):
    model = tf.keras.models.load_model(
        "./Data/" + name_prefix + "model_" + city + ".keras"
    )
    df = pd.read_pickle("./Data/prediction_data_cleaned_" + city + ".pkl")
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df.rename(
        columns={
            "day": "day_of_year",
            "tmax_avg": "tmax",
            "tmin_avg": "tmin",
            "prec_om": "prec",
            "humi_vc": "humi",
        }
    )
    # df.info()
    features = ["day_of_year", "tmax", "tmin", "prec", "humi"]
    df = df[features]
    target = "tmax"

    # df = df.iloc[:pd.to_datetime(end_date)]

    # Normalize the features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])

    # Use this many days of data to predict the next day's 'tmax'
    # X, y = create_dataset(df_scaled, df_scaled[:, 1], time_steps)
    # split = int(len(X) * 0.75)  # 70% for training

    # # Split the data
    # X_train, X_test = X[:split], X[split:]
    # y_train, y_test = y[:split], y[split:]
    old_data = df[-(time_steps):]
    if offset != 0:
        old_data = df[-(time_steps + offset) : -offset]
    # old_data.fillna(old_data.mean(), inplace=True)
    old_data.fillna(method="ffill", inplace=True)

    last_days_data = np.array(old_data)
    # print(last_days_data)
    last_days_scaled = scaler.transform(last_days_data)
    last_days_scaled = np.expand_dims(last_days_scaled, axis=0)
    predicted_tmax_scaled = model.predict(last_days_scaled, verbose=0)
    dummy_array = np.zeros((1, len(features)))
    dummy_array[:, 1] = predicted_tmax_scaled
    inverse_transformed_array = scaler.inverse_transform(dummy_array)
    predicted_tmax = inverse_transformed_array[:, 1]

    # print(f"Predicted 'tmax' for {city} for next day: {predicted_tmax[0]}")
    return predicted_tmax[0]


start_date = "2024-03-24"
end_date = "2024-04-01"
today = datetime.now()
# Format the date
today_filter = today.strftime("%Y-%m-%d")
# end_date = today_filter


# ==================================================================================
# IMPORTANT: UNCOMMENT THIS TO ACTUALLY GET DAILY DATA
# ==================================================================================
getDailyData(start_date, end_date)
prediction_results = []
offset = 0

for offset in range(1):
    for city in cities:
        from datetime import timedelta

        print(f"----------- {city} -----------")
        print(f"=========== {today_filter} ===========")
        pred_date = pd.to_datetime(end_date, format="%Y-%m-%d") + timedelta(days=1)
        #   pred_date = end_date
        pred = getPrediction(city)
        prediction_results.append(
            {"date": str(pred_date)[:10], "city": city, "tmax_predicted": pred}
        )
df_predictions = pd.DataFrame(prediction_results)
df_predictions.to_csv("predictions_final.csv", mode="a", header=False)
pprint.pp(prediction_results)
