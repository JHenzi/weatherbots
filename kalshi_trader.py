import uuid
import kalshi_python
from kalshi_python.models import *
import pprint
from datetime import datetime
import pandas as pd

# Add code for Kalshi

config = kalshi_python.Configuration()
# Comment the line below to use production
config.host = 'https://demo-api.kalshi.co/trade-api/v2'

# Create an API configuration passing your credentials.
# Use this if you want the kalshi_python sdk to manage the authentication for you.
kalshi_api = kalshi_python.ApiInstance(
    email='pgoyanka@bu.edu',
    password='***REMOVED***',
    configuration=config,
)

seriesTickers = ['HIGHNY', 'HIGHCHI', 'HIGHAUS', 'HIGHMIA']
latitude = [40.79736, 41.78701, 30.1444, 25.7738]
longitude = [-73.97785, -87.77166, -97.66876, -80.1936]
cities = ["ny", "il", "tx", "fl"]
stations_ncei = ["USW00094728", "USW00014819", "USW00013904", "USC00086315"]
time_steps = 60


def getTempRanges(seriesTicker):
  current_date = datetime.now()
  formatted_date = current_date.strftime("%y%b%d").upper()
  formatted_date


  eventTicker = seriesTicker + '-' + formatted_date
  eventResponse = kalshi_api.get_event(eventTicker)

  print ("Tempratue ranges for " + seriesTicker[4:] + " for date " + formatted_date)
  for temps in eventResponse.markets:
    print(temps.subtitle)
  return eventResponse

# for seriesTicker in seriesTickers:
#   temps = getTempRanges(seriesTicker)

# Function to determine the interval
def find_interval(value, intervals):
    for i in range(len(intervals)):
        if 'or below' in intervals[i].subtitle:
            max_value = float(intervals[i].subtitle.split('°')[0])
            if value <= max_value:
                return i
        elif 'or above' in intervals[i].subtitle:
            min_value = float(intervals[i].subtitle.split('°')[0])
            if value >= min_value:
                return i
        else:
            bounds = intervals[i].subtitle.replace('°', '').split(' to ')
            min_value, max_value = map(float, bounds)
            if min_value <= value <= max_value:
                return i
    return 0


def makeTrade(pred, eventsResponse):
  # Check the interval for pred
  interval = find_interval(pred, eventsResponse.markets)

  print(f"The value {pred} falls into the interval: '{eventsResponse.markets[interval].subtitle}'")

  print(eventsResponse.markets[interval])


  ticker = eventsResponse.markets[interval].ticker

  order_params = {'ticker':ticker,
                      'client_order_id':str(uuid.uuid4()),
                      'type':'limit',
                      'action':'buy',
                      'side':'no',
                      'count':10,
                      'yes_price':100,
                      'no_price':30,
                      'expiration_ts':None,
                      'sell_position_floor':None,
                      'buy_max_cost':None}

  # Checks if the exchange is available.
  exchangeStatus = kalshi_api.get_exchange_status()
  print('Exchange status response: ')
  # pprint.pprint(exchangeStatus)


  # Gets the balance for your kalshi account.
  balanceResponse = kalshi_api.get_balance()
  print('\nUser balance: ')
  pprint.pprint(balanceResponse)

  if exchangeStatus.trading_active:
      # Submit an order for 10 yes contracts at 50cents on 'FED-23DEC-T3.00'.
      orderUuid = str(uuid.uuid4())
      orderResponse = kalshi_api.create_order(CreateOrderRequest(
          ticker=eventsResponse.markets[interval].ticker,
          action='buy',
          type='limit',
          yes_price=50,
          count=10,
          client_order_id=orderUuid,
          side='yes',
      ))
      print('\nOrder submitted: ', eventsResponse.markets[interval].ticker)
      # pprint.pprint(orderResponse)
  else:
      print('\nThe exchange is not trading active, no orders will be sent right now.')

for i in range(len(seriesTickers)):
  print(f"----------- {seriesTickers[i]} -----------")
  eventsResponse = getTempRanges(seriesTickers[i])
  df = pd.read_csv("predictions_final.csv")
  today = datetime.now()
  # Format the date
  today_filter = today.strftime('%Y-%m-%d')
  print(f"=========== {today_filter} ===========")
  filtered_df = df[(df['date'] == today_filter) & (df['city'] == cities[i])]
  print(filtered_df)
  pred = float(filtered_df['tmax_predicted'])
  makeTrade(pred, eventsResponse)
