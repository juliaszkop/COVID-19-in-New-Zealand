import pandas as pd
from utils import covid_forecast_arima, save_or_show_plotly

df = pd.read_csv('../data/New_Zealand_covid_data_cleaned.csv')
fig, forecast_df, model = covid_forecast_arima(df, target_variable='new_cases', train_start='2022-01-01', train_end='2023-01-01', forecast_from='2023-01-01', periods=600)

fig.savefig('../images/New_Zealand_actual_vs_forecast_from_2022.png', dpi = 300)
