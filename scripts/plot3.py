import pandas as pd
from utils import create_period_line_plot, save_or_show_plotly

df = pd.read_csv('../data/New_Zealand_covid_data_cleaned.csv')
df['date'] = pd.to_datetime(df['date'])
fig = create_period_line_plot(df)
save_or_show_plotly(fig, 'New_Zealand_period_line_plot.html')
