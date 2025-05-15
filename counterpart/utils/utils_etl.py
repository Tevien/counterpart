import os
import dask.dataframe as dd
import polars as pl
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('luigi-interface')

def first_non_nan(x):
    return x[np.isfinite(x)][0]

def convert_bytes_to_mb(num):
    """
    this function will convert bytes to MB
    """
    num /= 1024.0**2
    print(num)
    return num


def file_size(file_path):
    """
    this function will return the file size
    """
    file_info = os.stat(file_path)
    print (file_path)
    return convert_bytes_to_mb(file_info.st_size)

def return_subset(filename, cols, index_col=None, blocksize=10000):
    """
    this function will return a subset of the dataframe

    Args:
    filename: str
        The filename of the dataframe
    cols: list
        The columns to return
    index_col: str
        The index column
    blocksize: int
        The blocksize to use
    """
    # Is the file a parquet file?
    if '.parquet' in filename:
        df = dd.read_parquet(filename, columns=cols+[index_col])
    elif '.feather' in filename:
        df = dd.from_pandas(pd.read_feather(filename, columns=cols+[index_col]), npartitions=3)
    else:
        df = dd.read_csv(filename, blocksize=blocksize)
        df = df.loc[:, cols+[index_col]]

    if index_col is not None:
        df = df.set_index(index_col)
    return df

def vals_to_cols(filename, cols, index_col=None, blocksize=10000, agg=None):
    col_name = cols[0]
    valname = list(cols[1].keys())[0]
    vars = cols[1][valname]
    try:
        k_vars = [int(list(v.keys())[0]) for v in vars]
    except ValueError:
        k_vars = [list(v.keys())[0] for v in vars]
    
    if '.parquet' in filename:
        df = dd.read_parquet(filename)
    elif '.feather' in filename:
        df = dd.from_pandas(pd.read_feather(filename), npartitions=3)
    else:
        df = dd.read_csv(filename, blocksize=blocksize)

    # Filter col name on interested vars
    df_filtered = df.loc[df[col_name].isin(k_vars)]

    #dask requires string as category type
    df_filtered[col_name] = df_filtered[col_name].astype('str')
    df_filtered[col_name] = df_filtered[col_name].astype('category')
    df_filtered[col_name] = df_filtered[col_name].cat.as_known()

    if not agg:
        logging.info(f"index col {index_col}, col name {col_name}, valname {valname}, k_vars {k_vars}")
        #fnn = dd.Aggregation(name='fnn', chunk=lambda x: x[np.isfinite(x)][0], agg=lambda x: x[np.isfinite(x)][0])
        df_pivoted = df_filtered.pivot_table(index=index_col, columns=col_name, values=valname, 
                                            aggfunc='last').compute()
        df_pivoted_first = df_filtered.pivot_table(index=index_col, columns=col_name, values=valname, 
                                            aggfunc='first').compute()
        df_pivoted_count = df_filtered.pivot_table(index=index_col, columns=col_name, values=valname, 
                                            aggfunc='count').compute()
        # Check if count is more than 2
        df_pivoted_count = df_pivoted_count.where(df_pivoted_count > 2, other=0)
        df_pivoted_count = df_pivoted_count.mask(df_pivoted_count > 2, other=1)
        print(f"Number of entries where count is more than 2: {df_pivoted_count.sum().sum()}")
        # TODO: Make this more robust, e.g. if there are more than 2 entries, then take the first non-nan value
        # This was done with speed in mind

        df_pivoted = df_pivoted.fillna(df_pivoted_first)
    else:
        df_pivoted_count = df_filtered.pivot_table(index=index_col, columns=col_name, values=valname, 
                                            aggfunc=agg).compute()

    
    # Combine vars to dict
    d_vars = {list(v.keys())[0]: list(v.values())[0] for v in vars}

    # Rename column
    df_pivoted = df_pivoted.rename(columns=d_vars)
    cols_dict = {col: str(col) for col in df_pivoted.columns if not isinstance(col, str)}
    df_pivoted = df_pivoted.rename(columns=cols_dict)
    
    return df_pivoted

def checkpoint(_df, _filename):
    """
    this function will checkpoint the dataframe to a parquet file
    """
    _df.to_parquet(_filename, engine='pyarrow', compression='snappy')
    return _filename