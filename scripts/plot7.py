import pandas as pd
from utils import create_gender_pie_chart, save_or_show_plotly

df = pd.read_csv('../data/covid-case-counts.csv')
fig = create_gender_pie_chart(df)

save_or_show_plotly(fig, 'NZ_cases_by_gender_pie_chart.html')