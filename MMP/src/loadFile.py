import os
import pandas as pd

def clear_directory(directory):
    if os.path.exists(directory):
        for file in os.listdir(directory):
            os.remove(os.path.join(directory, file))

    os.makedirs(
        directory,
        exist_ok=True
    )

def load_and_process_file(file_path, week):

    if not os.path.exists(file_path):
        raise ValueError(
            f'Файл не найден: {file_path}'
        )

    df = pd.read_excel(file_path)

    df['time_dt'] = pd.to_datetime(df['time_dt'])
    df['week'] = week

    base_cols = ['time_dt', 'week', 'stocks', 'futures']
    price_qty_cols = [col for col in df.columns if 'price0' in col or 'quantity0' in col]
    columns_to_keep = base_cols + price_qty_cols

    df = df[columns_to_keep]

    rename_dict = {
        'OFFERfutures_price0': 'OFFER_F_P0',
        'OFFERfutures_quantity0': 'OFFER_F_Q0',
        'BIDfutures_price0': 'BID_F_P0',
        'BIDfutures_quantity0': 'BID_F_Q0',
        'OFFERstocks_price0': 'OFFER_S_P0',
        'OFFERstocks_quantity0': 'OFFER_S_Q0',
        'BIDstocks_price0': 'BID_S_P0',
        'BIDstocks_quantity0': 'BID_S_Q0',
    }

    df = df.rename(columns=rename_dict)

    return df


def return_files(folder_path):
    return [
        os.path.join(folder_path, 'TRNFP-23_12-28_12-with_duplicates.xlsx'),
        os.path.join(folder_path, 'TRNFP-30_12-10_01-with_duplicates.xlsx'),
        os.path.join(folder_path, 'TRNFP-13_01-17_01-with_duplicates.xlsx'),
        os.path.join(folder_path, 'TRNFP-20_01-24_01-with_duplicates.xlsx'),
        os.path.join(folder_path, 'TRNFP-27_01-31_01-with_duplicates.xlsx'),
        os.path.join(folder_path, 'TRNFP-03_02-14_02-with_duplicates.xlsx'),
        os.path.join(folder_path, 'TRNFP-17_02-28_02-with_duplicates.xlsx'),
        os.path.join(folder_path, 'TRNFP-03_03-14_03-with_duplicates.xlsx'),
    ]


def make_df_all():
    files = return_files('./data/')

    df_list = []
    for i, file in enumerate(files):
        df_temp = load_and_process_file(
            file,
            week=i
        )
        df_list.append(df_temp)

    df_all = pd.concat(
        df_list,
        ignore_index=True
    )

    df_all = df_all.sort_values('time_dt').reset_index(drop=True)

    return df_all