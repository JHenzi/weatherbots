import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = "./Kalshi-Recent-Activity (1).csv"
data = pd.read_csv(file_path)

# Convert 'Settled_Time' to datetime format
data["Settled_Time"] = pd.to_datetime(data["Settled_Time"])

# Filter the dataset to include only settlements
settlements = data[data["type"] == "Settlement"]

# Further split into profitable and non-profitable trades
profitable = settlements[settlements["Profit_In_Dollars"] > 0]
non_profitable = settlements[settlements["Profit_In_Dollars"] <= 0]

# Ensure the dates are sorted in ascending order before plotting
profitable_sorted = profitable.sort_values(by="Settled_Time")
non_profitable_sorted = non_profitable.sort_values(by="Settled_Time")

# Plot with dates in ascending order
plt.figure(figsize=(14, 7))

# Plot profitable trades with formatted and sorted 'Settled_Time' on the X-axis
sns.scatterplot(
    x=profitable_sorted["Settled_Time"].dt.strftime("%d-%b"),
    y=profitable_sorted["Profit_In_Dollars"],
    color="green",
    label="Profitable",
)

# Plot non-profitable trades with formatted and sorted 'Settled_Time' on the X-axis
sns.scatterplot(
    x=non_profitable_sorted["Settled_Time"].dt.strftime("%d-%b"),
    y=non_profitable_sorted["Profit_In_Dollars"],
    color="red",
    label="Non-Profitable",
)

plt.title("Profit and Loss from Trades Over Time")
plt.xlabel("Trade Date (Day-Month)")
plt.ylabel("Profit / Loss in Dollars")
plt.legend()
plt.grid(True)

# Improve readability of the date labels on the X-axis
plt.xticks(rotation=45)
plt.tight_layout()

# plt.show()

# Group the settlement data by the date part of 'Settled_Time'
daily_summary = (
    settlements.groupby(settlements["Settled_Time"].dt.date)
    .agg(
        Total_Trades=("Profit_In_Dollars", "count"),
        Total_Profit_Loss=("Profit_In_Dollars", "sum"),
        Average_Profit_Loss_Per_Trade=("Profit_In_Dollars", "mean"),
    )
    .reset_index()
)

# Rename 'Settled_Time' to 'Date' for clarity
daily_summary.rename(columns={"Settled_Time": "Date"}, inplace=True)

daily_summary

daily_summary.to_csv("Daily-Summary.csv")

import re


# Function to extract city name from the 'Market_Title' column
def extract_city(title):
    # Attempt to find a city name using a regular expression pattern
    match = re.search(r"in\s+([A-Za-z\s]+)", title)
    if match:
        return match.group(1).strip()
    return "Unknown"  # Default value if city name isn't found


# Apply the function to create a new 'City' column
settlements["City"] = settlements["Market_Title"].apply(extract_city)

# Group the data by date and city, then aggregate
city_daily_summary = (
    settlements.groupby([settlements["Settled_Time"].dt.date, "City"])
    .agg(
        Total_Trades=("Profit_In_Dollars", "count"),
        Total_Profit_Loss=("Profit_In_Dollars", "sum"),
        Average_Profit_Loss_Per_Trade=("Profit_In_Dollars", "mean"),
    )
    .reset_index()
)

city_daily_summary.to_csv("Daily-Summary.csv")
