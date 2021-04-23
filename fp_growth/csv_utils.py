import pandas as pd


def get_flat_file_data_frame(url):
    dataframe = pd.read_csv(url, keep_default_na=False,
                            encoding='utf-8', chunksize=1)
    return None
