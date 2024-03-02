# cs542-common-task

Kalshi Weather Prediction Common Task for BU's CS542 Spring 2024

# Reference Papers

- https://arxiv.org/abs/2309.13330

# Data Sources

- National Weather Data: https://www.weather.gov/documentation/services-web-api
- Open-Meteo

# Zip Codes

- (Belvedere Castle) Central Park, New York - 10024
  40.79736,-73.97785
- Midway International Airport, Chicago, Illinois - 60638
  41.78701,-87.77166
- Bergstrom International Airport, Austin, Texas - 78719
  30.14440,-97.66876
- Miami, Florida - 33101
  25.77380,-80.19360

  25.77380,-80.19360
  30.14440,-97.66876
  41.78701,-87.77166
  40.79736,-73.97785

# To Do

- [x] Clean up data fetcher script
- [x] Store everything as a CSV
- [x] Clean the data
- [x] Run a basic linear regression model
- [ ] Check predictions

# Notes

## March 2, 2024
Clearly, just running a linear regression using time/date as a feature doesn't work. Cuz it just keeps decreasing the temp assuming its a linear function.

Need to use more features.

Possible features:
- Humidity
- Sunshine Duration
- precipitation_probability_max
