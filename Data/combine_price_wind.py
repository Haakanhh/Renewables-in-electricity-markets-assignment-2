import pandas as pd
from pathlib import Path
import itertools

base = Path(__file__).resolve().parent.parent  # since script is in Data/
price_sample_path = base / "Data" / "random_20days_price_sample.csv"
wind_sample_path = base / "Data" / "random_20days_wind_sample.csv"
out_path = base / "Data" / "combined_price_wind_revenue.csv"

# Read the samples
columns = ['sample_type', 'date', 'HourUTC', 'HourDK', 'PriceArea', 'SpotPriceDKK', 'SpotPriceEUR', 'time', 'local_time', 'electricity']
price_df = pd.read_csv(price_sample_path, header=None, names=columns)
price_df['HourUTC'] = pd.to_datetime(price_df['HourUTC'], errors='coerce')
price_df['HourDK'] = pd.to_datetime(price_df['HourDK'], errors='coerce')
price_df['date'] = pd.to_datetime(price_df['date'], errors='coerce')
price_df['hour'] = price_df['HourDK'].dt.hour

wind_df = pd.read_csv(wind_sample_path, header=None, names=columns)
wind_df['time'] = pd.to_datetime(wind_df['time'], errors='coerce')
wind_df['local_time'] = pd.to_datetime(wind_df['local_time'], errors='coerce')
wind_df['date'] = pd.to_datetime(wind_df['date'], errors='coerce')
wind_df['hour'] = wind_df['local_time'].dt.hour

# Get unique dates
price_dates = sorted(price_df['date'].unique())
wind_dates = sorted(wind_df['date'].unique())

print(f"Price dates: {len(price_dates)}, Wind dates: {len(wind_dates)}")

# Prepare output list
results = []

combined_day = 0
for price_date, wind_date in itertools.product(price_dates, wind_dates):
    combined_day += 1
    for hour in range(24):
        # Get price for this date and hour
        price_row = price_df[(price_df['date'] == price_date) & (price_df['hour'] == hour)]
        if price_row.empty:
            continue  # skip if no data
        price = float(price_row['SpotPriceEUR'].iloc[0])
        
        # Get wind for this date and hour
        wind_row = wind_df[(wind_df['date'] == wind_date) & (wind_df['hour'] == hour)]
        if wind_row.empty:
            continue
        electricity_kW = float(wind_row['electricity'].iloc[0])
        
        # Revenue: electricity in MW * price in EUR/MWh = EUR
        revenue = (electricity_kW / 1000) * price
        
        results.append({
            'combined_day': combined_day,
            'price_date': price_date.date(),
            'wind_date': wind_date.date(),
            'hour': hour,
            'electricity_kW': electricity_kW,
            'price_EUR_per_MWh': price,
            'revenue_EUR': revenue
        })

# Create DataFrame and save
output_df = pd.DataFrame(results)
output_df.to_csv(out_path, index=False)

print(f"Created {len(output_df)} rows in {out_path}")
print(f"Total combined days: {combined_day}")
