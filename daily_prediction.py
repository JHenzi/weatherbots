import argparse
import json
import os
import pathlib
import pprint
import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None

warnings.filterwarnings("ignore")

# Load API keys
from dotenv import load_dotenv

load_dotenv()
# print((os.getenv("VISUAL_CROSSING_API_KEY")))

# Define some constants
latitude = [40.79736, 41.78701, 30.1444, 25.7738]
longitude = [-73.97785, -87.77166, -97.66876, -80.1936]
cities = ["ny", "il", "tx", "fl"]
stations_ncei = ["USW00094728", "USW00014819", "USW00013904", "USC00086315"]
start_date = "2016-01-01"
# end_date = "2024-03-24"
time_steps = 10

# ## API Calls to collect historical weather data

# ### Open Meteo


def getDataFromOpenMeteo(latitude, longitude, startDate, endDate, fileName):
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

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
    import requests

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
    # Avoid dumping full responses to stdout (noisy, and can leak details).
    if response.status_code != 200:
        print(
            f"Visual Crossing error for {fileName} {startDate}→{endDate}: {response.status_code}"
        )
    return response


# ### Meteostat

def getDataFromMeteostat(latitude, longitude, startDate, endDate, fileName):
    from meteostat import Daily, Stations

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
    import requests

    base_url = "https://www.ncei.noaa.gov/access/services/data/v1?"
    # Define the query parameters
    params = {
        "dataset": "daily-summaries",
        "stations": station,
        "startDate": startDate,
        "endDate": endDate,
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
            city_data = city_history_dfs[i][
                [
                    "date",
                    "tmax_vc",
                    "tmax_om",
                    "tmax_ms",
                    "tmax_ncei",
                    "tmin_vc",
                    "tmin_om",
                    "humi_vc",
                    "prec_om",
                    "tmin_ms",
                    "tmin_ncei",
                ]
            ]
            # NOTE: Some sources can be missing for recent days; we compute averages over available columns.

            def cleanAndPreprocessData(df):
                # Open-Meteo + Meteostat temps are in °C; VisualCrossing + NCEI are already in °F.
                columns_to_convert_to_farhenheit = ["tmax_om", "tmin_om", "tmax_ms", "tmin_ms"]
                for column in columns_to_convert_to_farhenheit:
                    df[column] = df[column] * 9 / 5 + 32
                df["date"] = pd.to_datetime(df["date"])
                df["day"] = df["date"].dt.dayofyear
                df["tmax_avg"] = df[["tmax_vc", "tmax_om", "tmax_ms", "tmax_ncei"]].mean(axis=1)
                df["tmin_avg"] = df[["tmin_vc", "tmin_om", "tmin_ms", "tmin_ncei"]].mean(axis=1)
                return df

            city_data = cleanAndPreprocessData(city_data)
            city_data.to_pickle(outputFilePrefix + cities[i] + ".pkl")

    # cleanAllData()
    cleanAllData("./Data/prediction_merged_df_", "./Data/prediction_data_cleaned_")
    print("New data has been cleaned!")


def getPrediction(city, name_prefix="", offset=0):
    if tf is None:
        raise RuntimeError(
            "TensorFlow is not installed. Install it (e.g. `pip install tensorflow`) "
            "before running predictions."
        )
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
    df = df.ffill().bfill()
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
    old_data = old_data.ffill().bfill()

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


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch latest weather inputs, predict tmax, and write predictions CSV."
    )
    parser.add_argument(
        "--trade-date",
        type=str,
        default=None,
        help="YYYY-MM-DD date of the Kalshi event you plan to trade (default: today).",
    )
    parser.add_argument(
        "--fetch-date",
        type=str,
        default=None,
        help="YYYY-MM-DD date of observed weather to append (default: trade_date - 1).",
    )
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default="predictions_final.csv",
        help="Where to write predictions (default: predictions_final.csv).",
    )
    parser.add_argument(
        "--prediction-mode",
        type=str,
        default="lstm",
        choices=["lstm", "forecast", "blend"],
        help=(
            "How to produce tmax_predicted. "
            "lstm = model only; forecast = provider forecast only; blend = weighted mix."
        ),
    )
    parser.add_argument(
        "--blend-forecast-weight",
        type=float,
        default=0.8,
        help="When --prediction-mode=blend, weight on forecast (0..1). Default 0.8",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip data fetching/cleaning and only run prediction from existing pickles.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Tomorrow.io free tier limits (as noted by user):
    # - 3 requests/second
    # - 25 requests/hour
    # - 500 requests/day
    # We enforce a basic per-process throttle and a 1h on-disk cache to prevent repeated runs
    # from burning hourly quota.
    _tomorrow_state: dict[str, float | None] = {"last_req_ts": None}

    trade_dt = (
        datetime.strptime(args.trade_date, "%Y-%m-%d").date()
        if args.trade_date
        else date.today()
    )
    fetch_dt = (
        datetime.strptime(args.fetch_date, "%Y-%m-%d").date()
        if args.fetch_date
        else (trade_dt - timedelta(days=1))
    )
    # We need a *contiguous* recent window for the LSTM input. Fetch enough days ending at fetch_dt
    # so the last `time_steps` rows correspond to recent conditions (not stale history).
    lookback_days = max(time_steps + 2, 14)
    start_dt = fetch_dt - timedelta(days=lookback_days - 1)
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = fetch_dt.strftime("%Y-%m-%d")

    if not args.skip_fetch:
        # Fetch a recent observed window and regenerate prediction pickles.
        getDailyData(start_date, end_date)
    else:
        # Guardrail: if you skip fetch, you're using whatever is already in Data/*.pkl.
        # If that data is stale, you'll get "reasonable" outputs for the *wrong time of year*.
        try:
            df_check = pd.read_pickle("./Data/prediction_data_cleaned_ny.pkl")
            df_check["date"] = pd.to_datetime(df_check["date"])
            last_dt = df_check["date"].max().date()
            if trade_dt > last_dt + timedelta(days=1):
                print(
                    f"WARNING: --skip-fetch with trade_date={trade_dt} but latest data is {last_dt}. "
                    f"Predictions will NOT reflect current conditions. Run without --skip-fetch to fetch recent data."
                )
        except Exception:
            pass

    def forecast_tmax_open_meteo(city: str) -> float | None:
        """Forecast tmax (°F) for trade_dt from Open-Meteo forecast API."""
        import requests

        i = cities.index(city)
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude[i],
            "longitude": longitude[i],
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "timezone": "UTC",
            "start_date": trade_dt.strftime("%Y-%m-%d"),
            "end_date": trade_dt.strftime("%Y-%m-%d"),
        }
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return None
        js = r.json()
        daily = js.get("daily") or {}
        temps = daily.get("temperature_2m_max") or []
        if not temps:
            return None
        return float(temps[0])

    def forecast_tmax_visual_crossing(city: str) -> float | None:
        """Forecast tmax (°F) for trade_dt from Visual Crossing timeline API."""
        import requests

        api_key = os.getenv("VISUAL_CROSSING_API_KEY")
        if not api_key:
            return None
        i = cities.index(city)
        start = trade_dt.strftime("%Y-%m-%d")
        end = trade_dt.strftime("%Y-%m-%d")
        url = (
            "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
            + str(latitude[i])
            + "%2C"
            + str(longitude[i])
            + "/"
            + start
            + "/"
            + end
            + "?unitGroup=us&include=days&key="
            + api_key
            + "&contentType=json"
        )
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None
        js = r.json()
        days = js.get("days") or []
        if not days:
            return None
        # 'tempmax' is in °F because unitGroup=us
        tm = days[0].get("tempmax")
        return float(tm) if tm is not None else None

    def forecast_tmax_tomorrow(city: str) -> float | None:
        """Forecast tmax (°F) for trade_dt from Tomorrow.io forecast API."""
        import time
        import requests
        import requests_cache

        api_key = os.getenv("TOMORROW")
        if not api_key:
            return None
        i = cities.index(city)
        location = f"{latitude[i]},{longitude[i]}"
        url = "https://api.tomorrow.io/v4/weather/forecast"
        params = {
            "location": location,
            "timesteps": "1d",
            "units": "imperial",
            "apikey": api_key,
        }
        # 1-hour cache to avoid repeat calls (protects 25 req/hour limit).
        # Stored in workspace so it survives across runs.
        session = requests_cache.CachedSession("Data/tomorrow_cache", expire_after=3600)

        # Throttle non-cached requests to stay under 3 req/sec.
        last_ts = _tomorrow_state.get("last_req_ts")
        if last_ts is not None:
            elapsed = time.time() - float(last_ts)
            if elapsed < 0.40:
                time.sleep(0.40 - elapsed)

        r = session.get(url, params=params, timeout=30)
        if not getattr(r, "from_cache", False):
            _tomorrow_state["last_req_ts"] = time.time()
        if r.status_code != 200:
            return None
        js = r.json()
        tl = (js.get("timelines") or {}).get("daily") or []
        if not tl:
            return None
        target = trade_dt.strftime("%Y-%m-%d")
        for item in tl:
            t = (item.get("time") or "")
            if t.startswith(target):
                vals = item.get("values") or {}
                v = vals.get("temperatureMax")
                return float(v) if v is not None else None
        # fallback: if first daily item is trade date
        first_time = (tl[0].get("time") or "")
        if first_time.startswith(target):
            vals = tl[0].get("values") or {}
            v = vals.get("temperatureMax")
            return float(v) if v is not None else None
        return None

    def forecast_tmax_weatherapi(city: str) -> float | None:
        """Forecast tmax (°F) for trade_dt from WeatherAPI.com forecast endpoint."""
        import requests_cache
        import requests

        api_key = os.getenv("WEATHERAPI")
        if not api_key:
            return None

        i = cities.index(city)
        q = f"{latitude[i]},{longitude[i]}"
        url = "https://api.weatherapi.com/v1/forecast.json"
        params = {
            "key": api_key,
            "q": q,
            # Forecast API requires 'days'; 'dt' restricts output to a specific date (must be within next 14 days).
            "days": 1,
            "dt": trade_dt.strftime("%Y-%m-%d"),
            "alerts": "no",
            "aqi": "no",
        }

        # Cache responses to reduce quota usage across repeated runs.
        session = requests_cache.CachedSession("Data/weatherapi_cache", expire_after=3600)
        r = session.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return None
        js = r.json()
        fc = (js.get("forecast") or {}).get("forecastday") or []
        if not fc:
            return None
        day = (fc[0].get("day") or {})
        v = day.get("maxtemp_f")
        return float(v) if v is not None else None

    prediction_results = []
    for city in cities:
        print(f"----------- {city} -----------")
        print(f"=========== trade_date={trade_dt} (fetch_date={fetch_dt}) ===========")
        pred_lstm = None
        # Always try to compute LSTM for logging/comparison if it's available.
        # In forecast-only mode, failure to compute the LSTM should not fail the run.
        if args.prediction_mode in ("lstm", "blend"):
            pred_lstm = getPrediction(city)
        else:
            try:
                pred_lstm = getPrediction(city)
            except Exception:
                pred_lstm = None

        pred_forecast = None
        forecast_sources: list[str] = []
        if args.prediction_mode in ("forecast", "blend"):
            om = forecast_tmax_open_meteo(city)
            vc = forecast_tmax_visual_crossing(city)
            tm = forecast_tmax_tomorrow(city)
            wa = forecast_tmax_weatherapi(city)
            vals = []
            if om is not None:
                vals.append(om)
                forecast_sources.append("open-meteo")
            if vc is not None:
                vals.append(vc)
                forecast_sources.append("visual-crossing")
            if tm is not None:
                vals.append(tm)
                forecast_sources.append("tomorrow")
            if wa is not None:
                vals.append(wa)
                forecast_sources.append("weatherapi")
            pred_forecast = float(np.mean(vals)) if vals else None

        if args.prediction_mode == "lstm":
            pred_final = float(pred_lstm)
        elif args.prediction_mode == "forecast":
            if pred_forecast is None:
                raise RuntimeError(
                    f"No forecast sources available for {city} on {trade_dt}. "
                    f"Check VISUAL_CROSSING_API_KEY and network access."
                )
            pred_final = float(pred_forecast)
        else:
            # blend
            if pred_forecast is None:
                pred_final = float(pred_lstm)
            else:
                w = float(args.blend_forecast_weight)
                pred_final = float(w * pred_forecast + (1 - w) * float(pred_lstm))

        prediction_results.append(
            {
                "date": trade_dt.strftime("%Y-%m-%d"),
                "city": city,
                "tmax_predicted": pred_final,
                "tmax_lstm": (None if pred_lstm is None else float(pred_lstm)),
                "tmax_forecast": (None if pred_forecast is None else float(pred_forecast)),
                "forecast_sources": ",".join(forecast_sources),
            }
        )

    df_predictions = pd.DataFrame(prediction_results)
    write_header = not os.path.exists(args.predictions_csv)
    df_predictions.to_csv(args.predictions_csv, mode="a", header=write_header, index=False)
    pprint.pp(prediction_results)
