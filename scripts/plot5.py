import pandas as pd
from utils import prepare_weekly_data_for_animation, animate_covid_barchart

df = prepare_weekly_data_for_animation([
    "../data/Poland_covid_data_cleaned.csv",
    "../data/New_Zealand_covid_data_cleaned.csv"
])

animate_covid_barchart(df)
