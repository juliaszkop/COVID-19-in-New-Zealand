import pandas as pd
from utils import create_vaccination_hospitalization_plot, save_or_show_plotly

nz_df = pd.read_csv('../data/New_Zealand_covid_data_cleaned.csv')
fig = create_vaccination_hospitalization_plot(nz_df)
save_or_show_plotly(fig, 'New_Zealand_vacc_hosp_plot.html')