import os
import sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from plotly.graph_objects import Figure, Scatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import matplotlib.animation as animation
from matplotlib.ticker import FuncFormatter
import seaborn as sns


def process_covid_data(country: str) -> None:
    os.makedirs(os.path.join("..", "data"), exist_ok=True)
    url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
    df = pd.read_csv(url)
    country_df = df[df['location'] == country].copy()
    country_df['date'] = pd.to_datetime(country_df['date'])
    country_df.fillna(method='ffill', inplace=True)
    country_df.fillna(method='bfill', inplace=True)
    country_df['new_cases_7day'] = country_df['new_cases'].rolling(window=7).mean()
    country_df['period'] = country_df['date'].apply(lambda x: '2020–2022' if x < pd.to_datetime('2023-01-01') else '2023–2024')
    safe_name = country.replace(' ', '_')
    filename = os.path.join("..", "data", f"{safe_name}_covid_data_cleaned.csv")
    country_df.to_csv(filename, index=False)
    print(f"Data for {country} has been saved to the file '{filename}'")

def covid_plot_new_cases_7days(df):
    fig = Figure()
    fig.add_trace(Scatter(x=df['date'], y=df['new_cases'], mode='lines', name='New cases', line=dict(color='#ffb6c1')))
    fig.add_trace(Scatter(x=df['date'], y=df['new_cases_7day'], mode='lines', name='7-day average', line=dict(color='#c71585')))
    fig.update_layout(title='COVID-19 New Cases', yaxis_title='Number of cases')
    return fig

def covid_plot_mortality_7days(df):
    fig = Figure()
    fig.add_trace(Scatter(x=df['date'], y=df['new_deaths'], mode='lines', name='death cases', line=dict(color='#CBA3E3')))
    fig.add_trace(Scatter(x=df['date'], y=df['new_deaths_smoothed'], mode='lines', name='7-day average', line=dict(color='#4B0082')))
    fig.update_layout(title='COVID-19 daily death cases', yaxis_title='Number of deaths')
    return fig

def save_or_show_plotly(fig, filename, mode=None):
    if mode is None:
        mode = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if mode == 1:
        fig.show()
    else:
        os.makedirs("../images", exist_ok=True)
        path = os.path.join("../images", filename)
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.png', '.jpg', '.jpeg'):
            width_px = int(12 * 100)
            height_px = int(10 * 100)
            fig.update_layout(width=width_px, height=height_px)
            pio.write_image(fig, path, width=width_px, height=height_px)
        else:
            if not path.endswith('.html'):
                path += '.html'
            pio.write_html(fig, path)
        print(f"Saved Plotly figure to {path}")

def create_vaccination_vs_cases_plot(df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(Scatter(x=df['date'], y=df['people_vaccinated_per_hundred'], name='Vaccinated (%)', line=dict(color= '#ffb6c1')), secondary_y=False)
    fig.add_trace(Scatter(x=df['date'], y=df['total_cases'], name='Total cases', line=dict(color='#ff1493' )), secondary_y=True)
    fig.update_layout(title='Vaccinations vs Total Cases', yaxis_title='Vaccinated (%)', yaxis2_title='Total cases')
    return fig

def create_period_line_plot(df):
    fig = px.line(df, x='date', y='new_cases_7day', color='period', title='New cases by epidemic periods', labels={'new_cases_7day': '7-day average'})
    return fig


def create_vaccination_hospitalization_plot(nz_df):
    nz_df['date'] = pd.to_datetime(nz_df['date'])
    cols = ['people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred', 'total_boosters_per_hundred', 'hosp_patients_per_million']
    nz_df[cols] = nz_df[cols].fillna(method='ffill')
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=nz_df['date'], y=nz_df['people_vaccinated_per_hundred'], name='At least 1 dose (%)', fill='tozeroy', mode='lines', line=dict(width=0.5, color='#ffb6c1')), secondary_y=False)
    fig.add_trace(go.Scatter(x=nz_df['date'], y=nz_df['people_fully_vaccinated_per_hundred'], name='Fully vaccinated (%)', fill='tozeroy', mode='lines', line=dict(width=0.5, color='#ff69b4')), secondary_y=False)
    fig.add_trace(go.Scatter(x=nz_df['date'], y=nz_df['total_boosters_per_hundred'], name='Booster dose (%)', fill='tozeroy', mode='lines', line=dict(width=0.5, color='#c71585')), secondary_y=False)
    fig.add_trace(go.Scatter(x=nz_df['date'], y=nz_df['hosp_patients_per_million'], name='Hospitalizations per million', mode='lines', line=dict(color='red', width=2.5)), secondary_y=True)
    fig.update_layout(title_text='<b>Impact of Vaccination on Hospitalizations in New Zealand</b><br><sup>Comparison of vaccination progress and hospital bed occupancy</sup>', xaxis_title='Date', legend_title='Legend', template='plotly_white', hovermode='x unified')
    fig.update_yaxes(title_text="<b>Population Vaccinated (%)</b>", secondary_y=False, color='darkblue')
    fig.update_yaxes(title_text="<b>Hospitalizations per Million</b>", secondary_y=True, color='red')
    return fig

def create_gender_pie_chart(df):
    df_filtered = df[df['Case Status'].isin(['Confirmed', 'Probable'])]
    df_filtered['Sex'] = df_filtered['Sex'].fillna('Unknown')
    gender_counts = df_filtered['Sex'].value_counts().reset_index()
    gender_counts.columns = ['Sex', 'Count']
    gender_counts['Percentage'] = (gender_counts['Count'] / gender_counts['Count'].sum() * 100).round(1)
    gender_counts['Label'] = gender_counts.apply(lambda row: f"{row['Sex']} ({row['Percentage']}%)", axis=1)
    fig = px.pie(gender_counts, names='Label', values='Count', title='Confirmed and probable COVID-19 cases by gender as of 3 April 2023', color='Sex', color_discrete_map={'Female': '#ffb6c1','Male': '#ff69b4','Unknown': '#c71585'})
    fig.update_traces(textinfo='none', marker=dict(line=dict(color='black', width=1)))
    fig.update_layout(title='Confirmed and probable COVID-19 cases by gender as of 3 April 2023', margin=dict(t=40, b=10, l=20, r=40), height=400, width=700, legend=dict(orientation='v', x=1, xanchor='center', yanchor='top'))
    return fig

def create_age_group_pie_chart(df):
    df_filtered = df[df['Case Status'].isin(['Confirmed', 'Probable'])]
    df_filtered['Age group'] = df_filtered['Age group'].fillna('Unknown')
    age_counts = df_filtered['Age group'].value_counts().reset_index()
    age_counts.columns = ['Age group', 'Count']
    age_order = ['0 to 9', '10 to 19', '20 to 29', '30 to 39', '40 to 49', '50 to 59', '60 to 69', '70 to 79', '80 to 89', '90+', 'Unknown']
    age_counts = age_counts[age_counts['Age group'].isin(age_order)]
    age_counts['Age group'] = pd.Categorical(age_counts['Age group'], categories=age_order, ordered=True)
    age_counts = age_counts.sort_values('Age group')
    age_counts['Percentage'] = (age_counts['Count'] / age_counts['Count'].sum() * 100).round(1)
    age_counts['Label'] = age_counts.apply(lambda row: f"{row['Age group']} ({row['Percentage']}%)", axis=1)
    label_order = age_counts['Label'].tolist()
    fig = px.pie(age_counts, names='Label', values='Count', title='Confirmed and probable COVID-19 cases by age group as of 3 April 2023', color='Label', category_orders={'Label': label_order}, color_discrete_sequence=px.colors.sequential.RdPu)
    fig.update_traces(textinfo='none', marker=dict(line=dict(color='black', width=1)))
    fig.update_layout(margin=dict(t=40, b=40, l=40, r=40), height=400, width=700, legend=dict(orientation='v', x=1, xanchor='left'))
    return fig

def load_covid_data(path):
    df = pd.read_csv(path, low_memory=False)
    df["Number of cases reported"] = pd.to_numeric(df["Number of cases reported"], errors="coerce").fillna(0)
    return df

def aggregate_dhb(df):
    return df.groupby("District", as_index=False)["Number of cases reported"].sum().rename(columns={"Number of cases reported": "cases"})

def map_to_regions(cases_by_dhb, mapping):
    rows = []
    for dhb, row in cases_by_dhb.set_index("District").iterrows():
        if dhb not in mapping:
            continue
        regions = mapping[dhb]
        per_region = row["cases"] / len(regions)
        for reg in regions:
            rows.append({"District": reg, "cases": per_region})
    return pd.DataFrame(rows).groupby("District", as_index=False)["cases"].sum()

def load_geo(json_path):
    gdf = gpd.read_file(json_path)
    return gdf.rename(columns={"name": "District"})

def merge_geo_cases(gdf, cases_by_geo):
    merged = gdf.merge(cases_by_geo, on="District", how="left")
    merged["cases"] = merged["cases"].fillna(0)
    return merged

def save_or_show_matplotlib(fig, filename, mode=None):
    if mode is None:
        mode = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if mode == 1:
        fig.show()
    else:
        os.makedirs("../images", exist_ok=True)
        path = os.path.join("../images", filename)
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved Matplotlib figure to {path}")

def create_choropleth(merged, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    merged.plot(column="cases", cmap="PuRd", linewidth=0.5, ax=ax, edgecolor="gray", legend=True, legend_kwds={"label": title, "shrink": 0.6})
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_axis_off()
    plt.tight_layout()
    return fig


def covid_forecast_arima(df, target_variable='new_cases_smoothed', train_start='2022-01-01', train_end='2023-12-31', forecast_from='2024-01-01', periods=90):
    df['date'] = pd.to_datetime(df['date'])
    ts = df[['date', target_variable]].copy().set_index('date')[target_variable]
    train = ts[train_start:train_end]
    try:
        model = SARIMAX(train, order=(7, 1, 2), seasonal_order=(1, 1, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
    except:
        model = ARIMA(train, order=(7, 1, 2))
        model_fit = model.fit()
    last_date = pd.to_datetime(forecast_from) - timedelta(days=1)
    forecast_dates = pd.date_range(start=forecast_from, periods=periods)
    forecast = model_fit.get_forecast(steps=periods)
    forecast_values = forecast.predicted_mean
    conf_int = forecast.conf_int()
    forecast_df = pd.DataFrame({'date': forecast_dates, 'forecast': forecast_values, 'lower_ci': conf_int.iloc[:, 0], 'upper_ci': conf_int.iloc[:, 1]}).set_index('date')
    plt.figure(figsize=(12, 6))
    plt.plot(ts.index, ts, label='Historical Data', color='purple')
    plt.plot(forecast_df.index, forecast_df['forecast'], label='Forecast', color='#c71585')
    plt.fill_between(forecast_df.index, forecast_df['lower_ci'], forecast_df['upper_ci'], color='#ffb6c1', alpha=0.3, label='95% CI')
    plt.axvline(x=last_date, color='gray', linestyle='--')
    plt.title(f'COVID-19 {target_variable.replace("_", " ").title()} Forecast')
    plt.xlabel('Date')
    plt.ylabel(target_variable.replace("_", " ").title())
    plt.legend()
    plt.grid(True)
    return plt.gcf(), forecast_df, model_fit

mutation_periods = [
("Alpha", pd.to_datetime("2020-11-01"), pd.to_datetime("2020-12-31"), "#cce5ff"),
("Beta", pd.to_datetime("2021-01-01"), pd.to_datetime("2021-03-31"), "#d0d0ff"),
("Kappa", pd.to_datetime("2021-04-01"), pd.to_datetime("2021-06-30"), "#e0d0ff"),
("Delta", pd.to_datetime("2021-07-01"), pd.to_datetime("2021-12-31"), "#ffe6cc"),
("Omicron", pd.to_datetime("2022-01-01"), pd.to_datetime("2023-06-30"), "#e6ffcc"),
("FLiRT", pd.to_datetime("2024-04-01"), pd.to_datetime("2025-06-06"), "#ccffaa"),
]


def prepare_weekly_data_for_animation(paths: list[str]) -> pd.DataFrame:
    
    all_data = []
    
    for path in paths:
        df = pd.read_csv(path, parse_dates=['date'])
        df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)
        country = df['location'].iloc[0]
        df_grouped = df.groupby('week').agg({
            'new_cases_7day': 'mean',
            'population': 'first'  
        }).reset_index()
        df_grouped['cases_per_million'] = df_grouped['new_cases_7day'] / df_grouped['population'] * 1e6
        df_grouped['country'] = country
        all_data.append(df_grouped[['week', 'country', 'cases_per_million']])
    
    return pd.concat(all_data)


def animate_covid_barchart(df: pd.DataFrame) -> None:
    sns.set(style="whitegrid")
    sns.set_palette("RdPu")

    weeks = sorted(df['week'].unique())
    countries = df['country'].unique()

    fig, ax = plt.subplots(figsize=(10, 6))
    max_value = df["cases_per_million"].max()
    def animate(i):
        ax.clear()
        current_week = weeks[i]
        weekly_data = df[df['week'] == current_week].copy()
        weekly_data['cases_per_million'] = weekly_data['cases_per_million'].fillna(0)
        weekly_data = weekly_data.sort_values("cases_per_million", ascending=False)

        colors = sns.color_palette("RdPu", len(weekly_data))
        bars = ax.bar(weekly_data['country'], weekly_data['cases_per_million'], color=colors)

        ax.set_ylim(bottom=0, top=max_value * 1.3)
        ax.set_ylabel("Cases per Million")
        ax.set_xlabel("Country")
        ax.set_title("Weekly COVID-19 Cases per Million", fontsize=16, fontweight='bold')
        ax.grid(False)

        
        for label, start, end, color in mutation_periods:
            if start <= current_week <= end:
                ax.axhspan(0, max_value * 1.3, facecolor=color, alpha=0.2)
                ax.text(0.5, 0.95, label,
                        transform=ax.transAxes, fontsize=14, color='black',
                        ha='center', va='top', fontweight='bold', alpha=0.7)

        
        for bar, val in zip(bars, weekly_data["cases_per_million"]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + max_value * 0.02,
                    f'{int(val):,}', ha='center', va='bottom', fontsize=11, fontweight='semibold')

        
        ax.text(0.95, 0.9, f"{current_week.strftime('%Y-%m-%d')}",
                transform=ax.transAxes, fontsize=16, color='black', ha='right', va='top',
                fontweight='bold')
    anim = animation.FuncAnimation(fig, animate, frames=len(weeks), interval=300, repeat=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.95)

    plt.show()
    anim.save("../images/covid_animation_mutatation.gif", writer="pillow", fps=5, dpi = 150)

def create_correlation_heatmap(df):
    lag_days = 7
    df['stringency_index_lag7'] = df['stringency_index'].shift(lag_days)
    df['new_vaccinations_smoothed_lag7'] = df['new_vaccinations_smoothed'].shift(lag_days)
    df['people_fully_vaccinated_per_hundred_lag7'] = df['people_fully_vaccinated_per_hundred'].shift(lag_days)
    core_cols = [
        'new_cases_smoothed', 'new_deaths_smoothed', 'total_cases_per_million',
        'total_deaths_per_million', 'reproduction_rate', 'new_tests_smoothed_per_thousand',
        'people_fully_vaccinated_per_hundred', 'total_boosters_per_hundred',
        'stringency_index_lag7', 'new_vaccinations_smoothed_lag7',
        'people_fully_vaccinated_per_hundred_lag7', 'excess_mortality_cumulative_per_million'
    ]
    df_core = df[core_cols].dropna()
    corr_matrix = df_core.corr()
    clean_labels = [col.replace('_', ' ').title() for col in corr_matrix.columns]
    corr_matrix.columns = clean_labels
    corr_matrix.index = clean_labels
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdPu", square=True, ax=ax)
    ax.set_title("Correlation Map", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig