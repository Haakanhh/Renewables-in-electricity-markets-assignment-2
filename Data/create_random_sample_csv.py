import random
from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parent
prices_path = base / "Data" / "DayAheadPrices_DK2.csv"
wind_path = base / "Data" / "ninja_wind_55.5783_15.7764_corrected.csv"
out_path = base / "Data" / "random_20days_price_wind_sample.csv"

random.seed(42) # Set seed for reproducibility

prices = pd.read_csv(prices_path, sep=';', decimal=',', parse_dates=['HourUTC', 'HourDK'])
prices['date'] = prices['HourDK'].dt.date
price_days = prices['date'].unique().tolist()
selected_price_days = random.sample(price_days, 20)
price_sample = prices[prices['date'].isin(selected_price_days)].copy()
price_sample = price_sample.assign(
    sample_type='price',
    time='',
    local_time='',
    electricity=''
)

wind = pd.read_csv(wind_path, comment='#', parse_dates=['time', 'local_time'])
wind['date'] = wind['local_time'].dt.date
wind_days = wind['date'].unique().tolist()
selected_wind_days = random.sample(wind_days, 20)
wind_sample = wind[wind['date'].isin(selected_wind_days)].copy()
wind_sample = wind_sample.assign(
    sample_type='wind',
    HourUTC='',
    HourDK='',
    PriceArea='',
    SpotPriceDKK='',
    SpotPriceEUR=''
)

output_columns = [
    'sample_type',
    'date',
    'HourUTC',
    'HourDK',
    'PriceArea',
    'SpotPriceDKK',
    'SpotPriceEUR',
    'time',
    'local_time',
    'electricity',
]

combined = pd.concat([price_sample[output_columns], wind_sample[output_columns]], ignore_index=True)
combined.to_csv(out_path, index=False)

print(f"Wrote {len(price_sample)} price rows and {len(wind_sample)} wind rows to {out_path}")
