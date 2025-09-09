import pandas as pd
from utils import covid_plot_mortality_7days, save_or_show_plotly

df = pd.read_csv('../data/New_Zealand_covid_data_cleaned.csv')
df['date'] = pd.to_datetime(df['date'])
fig = covid_plot_mortality_7days(df)
save_or_show_plotly(fig, 'NZ_covid_plot_mortality_7days.html')