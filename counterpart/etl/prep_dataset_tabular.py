from counterpart.utils.utils_etl import file_size, return_subset, vals_to_cols
from counterpart.utils.functions import transform_aggregations, merged_transforms
from counterpart.etl.convert_to_parquet import convert
import dask.dataframe as dd
from dask_ml.impute import SimpleImputer
from dask_ml.preprocessing import StandardScaler
from joblib import load, dump
import pandas as pd
import polars as pl
import numpy as np
import logging
import json
import luigi
import os
import snowflake.connector
from snowflake.connector.pandas_tools import pd_writer, write_pandas

logger = logging.getLogger('luigi-interface')

# meta data
__author__ = 'SB'
__date__ = '2025-05-10'

class PreProcess(luigi.Task):
    lu_output_path = luigi.Parameter(default='preprocessed.parquet')
    etl_config = luigi.Parameter(default="config/etl.json")
    
    def output(self):
        """
        Generates the output target for the ETL process.

        This method reads the ETL configuration file to determine the output
        directory for preprocessed data. If the 'preprocessing' key is not
        specified in the configuration, it defaults to a directory path
        constructed using the 'name' key from the configuration. The method
        ensures the output directory exists and returns a Luigi LocalTarget
        pointing to the specified output path.

        Returns:
            luigi.LocalTarget: A Luigi target object pointing to the output file.

        Raises:
            FileNotFoundError: If the ETL configuration file does not exist.
            KeyError: If the 'name' key is missing in the configuration and
                      'preprocessing' is not specified.
        """
        with open(self.etl_config, 'r') as f:
            input_json = json.load(f)
        outdir = input_json.get('preprocessing', None)
        if not outdir:
            name = input_json.get('name', None)
            outdir = os.path.join('data', name, 'preprocessed')
        os.makedirs(outdir, exist_ok=True)
        return luigi.LocalTarget(os.path.join(outdir, self.lu_output_path))
    
    def _connect_to_snowflake(self, snowflake_config):
        """Establish connection to Snowflake."""
        try:
            conn = snowflake.connector.connect(
                user=snowflake_config.get('user'),
                password=snowflake_config.get('password'),
                account=snowflake_config.get('account'),
                warehouse=snowflake_config.get('warehouse'),
                database=snowflake_config.get('database'),
                schema=snowflake_config.get('schema')
            )
            logger.info("Successfully connected to Snowflake")
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {str(e)}")
            raise
    
    def _fetch_data_from_snowflake(self, conn, query):
        """Fetch data from Snowflake using a SQL query."""
        try:
            logger.info(f"Executing Snowflake query: {query}")
            df = pd.read_sql(query, conn)
            logger.info(f"Query returned {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to execute Snowflake query: {str(e)}")
            raise
    
    def _process_snowflake_data(self, input_json):
        """Process data from Snowflake."""
        snowflake_config = input_json.get('snowflake', {})
        if not snowflake_config:
            logger.error("Snowflake configuration not found in config file")
            raise ValueError("Missing Snowflake configuration")
        
        # Connect to Snowflake
        conn = self._connect_to_snowflake(snowflake_config)
        
        # Fetch data using query from config
        query = snowflake_config.get('query')
        if not query:
            table = snowflake_config.get('table')
            if table:
                query = f"SELECT * FROM {table}"
            else:
                raise ValueError("Neither query nor table specified in Snowflake config")
        
        df = self._fetch_data_from_snowflake(conn, query)
        conn.close()
        
        # Convert to dask dataframe for further processing
        ddf = dd.from_pandas(df, npartitions=input_json.get('npartitions', 4))
        return ddf
    
    def run(self):
        """
        Executes the ETL process for preparing a tabular dataset.
        This method processes data from either a Snowflake database or local files,
        applies transformations, and saves the final dataset to a specified output location.
        Steps:
        1. Reads the ETL configuration from a JSON file.
        2. Determines the data source (Snowflake or local files).
        3. Processes data accordingly:
           - For Snowflake: Calls `_process_snowflake_data`.
           - For local files: Reads and optionally repartitions Parquet files.
        4. Iterates through specified filenames to preprocess and transform data:
           - Checks if preprocessed files exist; if so, loads them.
           - Otherwise, processes the files by extracting or transforming columns.
           - Applies any specified aggregations or transformations.
           - Saves intermediate results to checkpoint files.
           - Merges processed data into a single Dask DataFrame.
        5. Applies final transformations to the merged dataset.
        6. Reduces the dataset to the specified final columns.
        7. Saves the final dataset to the output location.
        Raises:
            AssertionError: If the index of a processed DataFrame is not unique.
        Notes:
            - The method uses Dask for handling large datasets.
            - Checkpoints intermediate results to avoid redundant processing.
            - Supports column extraction, transformations, and aggregations.
        Logging:
            - Logs information about processing steps, file handling, and transformations.
            - Provides debug information for intermediate and final DataFrame states.
        Output:
            - The final processed dataset is saved as a Parquet file.
        """
        with open(self.etl_config, 'r') as f:
            input_json = json.load(f)
        
        # Determine if we're using Snowflake or local files
        data_source = input_json.get('data_source', 'local')
        
        if data_source.lower() == 'snowflake':
            # Process data from Snowflake
            logger.info("Processing data from Snowflake")
            ddf = self._process_snowflake_data(input_json)
        else:
            # Original local file processing
            logger.info("Processing data from local files")
            df_path = input_json.get('tabular', None)
            npartitions = input_json.get('npartitions', None)
            ddf = dd.read_parquet(df_path, engine='pyarrow')
            
            if npartitions:
                ddf = ddf.repartition(npartitions=npartitions)

        filenames = input_json.keys()

        # Keep only names with normal extensions
        filenames = [f for f in filenames if any([f.endswith(ext) for ext in ['.csv', '.csv.gz', '.parquet', '.feather']])]
        path = input_json['absolute_path']
        filenames = [os.path.join(path, f) for f in filenames]

        # Open empty dask dataframe
        df_pp = None

        logging.info(f"Filenames: {filenames}")
        # Create the requested columns from each filename
        for o, f in zip(filenames, filenames):
            # First check if we have this file checkpointed
            current_name = f.split("/")[-1].split(".")[0]
            saved_loc = f"{input_json['preprocessing']}/{current_name}_preprocessed.parquet"
            # If file exists then load it
            if os.path.exists(saved_loc):
                print(f"*** Loading {saved_loc} ***")
                df = dd.read_parquet(saved_loc, npartitions=3) #TODO: Make this configurable
                print(df.head())
                if df_pp is None:
                    df_pp = df
                else:
                    print(df_pp.head())
                    df_pp = df_pp.merge(df, how="left")
                continue

            print(f"*** Processing {f} ***")
            vals = input_json[o.split("/")[-1]]
            index = vals[0]
            cols = vals[1:]

            # are any items in the list dictionaries?
            col_extract = not any([isinstance(v, dict) for v in vals])
            logger.info(f"*** Column extraction: {col_extract} ***")
            if col_extract:
                df = return_subset(f, cols, index_col=index)
            else:
                df = vals_to_cols(f, cols, index_col=index)
            assert df.index.unique, "Index is not unique"


            df_20 = df.head(20)
            logging.info("df pre transform:")
            logging.info(df_20)

            # See if any column names specify aggregations
            aggs = input_json.get('PreTransforms', None)
            if not aggs:
                logging.info("No aggregations or transforms specified")
            
            # Separate cols into dictionaries and non-dictionaries
            d_cols, l_cols = [], []
            for c in cols:
                if isinstance(c, dict):
                    d_cols.append(c)
                else:
                    l_cols.append(c)
            # Extract values from dictionaries in lists
            for d in d_cols:
                l_d = list(d.values())[0]
                for l in l_d:
                    l_cols.append(list(l.values())[0])
            print(f"Total cols: {l_cols}")

            # Find intersection of cols and aggs keys
            cols_for_aggregations = list(set(l_cols).intersection(aggs.keys()))

            # Apply aggregations
            if cols_for_aggregations:
                df = transform_aggregations(df, aggs, cols_for_aggregations)

            # Add new cols to the dataframe and join with subject_id
            #df_100 = df.head(100)
            #print(df_100)
            df_20 = df.head(20)
            logging.info("df post transform:")
            logging.info(df_20)
            # Checkpoint pre-concat
            # Save current transformed dataframe
            df.to_parquet(saved_loc)

            df_pp_20 = df.head(20)
            logging.info("df_pp premerge")
            logging.info(df_pp_20)
            if df_pp is not None:
                logging.info(f"Shape before merge: {df_pp.shape}")
            logging.info(f"New data shape: {df.shape}")
            if df_pp is None:
                df_pp = df
            else:
                df_pp = df_pp.merge(df, how="left")
            df_pp_20 = df.head(20)
            logging.info("df_pp postmerge")
            logging.info(df_pp_20)
            logging.info(f"Shape after merge: {df_pp.shape}")
        
        # Merged transforms
        end_transforms = input_json.get('MergedTransforms', None)
        df_pp = merged_transforms(df_pp, end_transforms)
        
        # Reduce to final specified columns
        df_pp = df_pp[input_json["final_cols"]]
        print(df_pp.head(20))
        print(f"Final shape: {df_pp.shape}")
        
        df_pp.to_parquet(self.output())


class ImputeScaleCategorize(luigi.Task):
    lu_output_path = luigi.Parameter(default='preprocessed_imputed.parquet')
    etl_config = luigi.Parameter(default="config/etl.json")

    def requires(self):
        return PreProcess(etl_config=self.etl_config)
    
    def output(self):
        with open(self.etl_config, 'r') as f:
            input_json = json.load(f)
        outdir = input_json.get('preprocessing', None)
        if not outdir:
            name = input_json.get('name', None)
            outdir = f"data/{name}/preprocessing"
        return luigi.LocalTarget(os.path.join(prefix, self.lu_output_path))
    
    def run(self):
        with open(self.etl_config, 'r') as f:
            input_json = json.load(f)
        name = input_json['name']

        # TODO: Fix for preprocessing locs
        assert all(x in list(input_json.keys()) for x in ["scaler", "imputer"]), "Scaler and imputer not specified"
        scaler_path = f"data/{name}/preprocessing/{input_json['scaler']}"
        imputer_path = f"data/{name}/preprocessing/{input_json['imputer']}"
        load_sc, load_imp = False, False
        if os.path.exists(scaler_path):
            load_sc = True
            scaler = load(scaler_path)
        if os.path.exists(imputer_path):
            load_imp = True
            imputer = load(imputer_path)
        
        ddf = dd.read_parquet(self.input())

        # Make cells with underscores nan
        to_nan = ["", "__", "_", "___"]
        ddf = ddf.mask(ddf.isin(to_nan), other=np.nan)

        # Find categorical columns
        total_cols = ddf.columns
        # Remove categorical columns from list
        categories = input_json['categories']
        num_cols = [c for c in total_cols if c not in categories.keys()]
        for n in num_cols:
            ddf[n] = ddf[n].astype('float32')

        # Scale numerical columns
        if not load_sc:
            scaler = StandardScaler()
            scaler.fit(ddf[num_cols])
        ddf[num_cols] = scaler.transform(ddf[num_cols])

        # Map categorical columns to binary if only 2
        for c in categories:
            cats = categories[c]
            if len(cats) == 2:
                map_c = {cats[0]: 0, cats[1]: 1}
                ddf[c] = ddf[c].map(map_c)
            else:
                # One hot encode categorical columns
                ddf = dd.get_dummies(ddf, columns=[c])
        
        # Impute missing values
        if not load_imp:
            imputer = SimpleImputer(strategy='median')
            imputer.fit(ddf)
        ddf = imputer.transform(ddf)

        # Save scaler/imputer to h5 file
        if not load_sc:
            dump(scaler, scaler_path)
        if not load_imp:
            dump(imputer, imputer_path)
        
        ddf.to_parquet(self.output().path)
        

if __name__ == '__main__':    
    luigi.build([PreProcess(), ImputeScaleCategorize()], workers=2, local_scheduler=True)
