# 1) See how many days were reported per year for the chosen station
from config import TOKEN, LAT, LON, UNITS
from noaa_api import find_nearby_stations, fetch_precip_for_years
from datetime import date

main_years = 2012,2018,2021,2024,2025
#sid = find_nearby_stations(TOKEN, LAT, LON, max_nearby=1)[0]
sid = 'GHCND:USW00014837'
for year in main_years:
    print("Station:", sid)
    end_doy = date.today().timetuple().tm_yday
    print("Reported days by year (out of", end_doy, "):")
    df = fetch_precip_for_years(TOKEN, sid, [year], UNITS)
    print(df.groupby("year")["doy"].nunique())