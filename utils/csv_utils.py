import pandas as pd


def get_flat_file_data_frame(url):
    """
    Description

    Args: 
        url (string): path to the csv file. 

    Returns: 
        data (ndarray): an n-dimensional array containing the data of the specified file. 
    """
    dataframe = pd.read_csv(url, keep_default_na=False,
                            encoding='utf-8', chunksize=1)
    return dataframe.index.as_list()
