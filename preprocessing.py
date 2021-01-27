import pandas as pd
from definitions import *


def process_electricity_loads(data_folder='data/UCI/'):
    data = pd.read_csv(data_folder + 'LD2011_2014.txt', sep=';', index_col=0, header=0)

    for col in data.columns:
        data[col] = data[col].astype(str)
        data[col] = data[col].str.replace(",",".")

    data = data.astype(float)
    data.index = pd.to_datetime(data.index)

    # Resample data by averaging daily consumption
    data_day = data.resample('D').sum()

    # Resample data by averaging hourly consumption
    data_hour = data.resample('H').sum()

    # Remove extreme values
    print(data_day.shape)
    cols = data_day.columns[data_day.mean() < 10 ** 4]
    print(data_day.shape)
    data_day = data_day[cols]
    data_hour = data_hour[cols]
    data_clean = data[cols]

    data_day.to_pickle(DATA_DIR + '/UCI/df_daily_electricity_loads.pkl')
    data_hour.to_pickle(DATA_DIR + '/UCI/df_hourly_electricity_loads.pkl')
    data_clean.to_pickle(DATA_DIR + '/UCI/df_clean_electricity_loads.pkl')
    print(data_day.shape, data_hour.shape, data_clean.shape)
    return data_day, data_hour


if __name__ == '__main__':
    process_electricity_loads()



