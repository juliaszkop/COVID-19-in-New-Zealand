import pandas as pd
from utils import (
    load_covid_data,
    aggregate_dhb,
    map_to_regions,
    load_geo,
    merge_geo_cases,
    create_choropleth,
    save_or_show_matplotlib
)

mapping = {
    "Auckland": ["Auckland"],
    "Waitemata": ["Auckland"],
    "Counties Manukau": ["Auckland"],
    "Capital & Coast/Hutt": ["Wellington"],
    "MidCentral": ["Manawatu-Wanganui"],
    "Whanganui": ["Manawatu-Wanganui"],
    "Wairarapa": ["Wellington"],
    "Tairawhiti": ["Gisborne District"],
    "Hawke's Bay": ["Hawke's Bay"],
    "Taranaki": ["Taranaki"],
    "Northland": ["Northland"],
    "Bay of Plenty": ["Bay of Plenty"],
    "Waikato": ["Waikato"],
    "Lakes": ["Bay of Plenty"],
    "Canterbury/West Coast": ["Canterbury", "West Coast"],
    "Southern": ["Otago", "Southland"],
    "Nelson Marlborough": ["Nelson City", "Marlborough District"],
    "South Canterbury": ["Canterbury"]
}

df_nz = load_covid_data("../data/covid-case-counts.csv")
cases_by_dhb = aggregate_dhb(df_nz)
cases_by_geo = map_to_regions(cases_by_dhb, mapping)
gdf_nz = load_geo("../data/nz.json")
merged_nz = merge_geo_cases(gdf_nz, cases_by_geo)
fig = create_choropleth(merged_nz, "COVID-19 Cases by NZ Region")
save_or_show_matplotlib(fig, "nz_choropleth.png")
