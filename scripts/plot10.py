import pandas as pd
from utils import create_correlation_heatmap, save_or_show_matplotlib

df = pd.read_csv('../data/New_Zealand_covid_data_cleaned.csv', parse_dates=['date'])
fig = create_correlation_heatmap(df)
save_or_show_matplotlib(fig, 'New_Zealand_correlation_heatmap.png')