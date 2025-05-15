from counterpart.utils.utils_etl import file_size, return_subset, vals_to_cols
from counterpart.utils.functions import transform_aggregations, merged_transforms
import logging
import json
import os
import dask.dataframe as dd

# Set up default logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# meta data
__author__ = 'SB'
__date__ = '2025-05-10'

class PreProcess:
    def __init__(self, snowpark_session=None, config_path="config/etl.json"):
        """
        Initialize the PreProcess class with a Snowpark session and configuration path.
        
        Args:
            snowpark_session: Snowpark session for connecting to Snowflake
            config_path: Path to the JSON configuration file
        """
        self.snowpark_session = snowpark_session
        self.config_path = config_path
        self.output_path = None
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
            
        # Set output path
        outdir = self.config.get('preprocessing', None)
        if not outdir:
            name = self.config.get('name', None)
            outdir = os.path.join('data', name, 'preprocessed')
        os.makedirs(outdir, exist_ok=True)
        self.output_path = os.path.join(outdir, self.config.get('output_file', 'preprocessed.parquet'))
    
    def _fetch_data_from_snowpark(self, query):
        """Fetch data from Snowflake using Snowpark session."""
        try:
            logger.info(f"Executing Snowpark query: {query}")
            snow_df = self.snowpark_session.sql(query)
            df = snow_df.to_pandas()
            logger.info(f"Query returned {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to execute Snowpark query: {str(e)}")
            raise
    
    def _process_snowpark_data(self):
        """Process data from Snowflake using Snowpark."""
        snowflake_config = self.config.get('snowflake', {})
        if not snowflake_config:
            logger.error("Snowflake configuration not found in config file")
            raise ValueError("Missing Snowflake configuration")
        
        # Fetch data using query from config
        query = snowflake_config.get('query')
        if not query:
            table = snowflake_config.get('table')
            if table:
                query = f"SELECT * FROM {table}"
            else:
                raise ValueError("Neither query nor table specified in Snowflake config")
        
        df = self._fetch_data_from_snowpark(query)
        
        # Convert to dask dataframe for further processing
        ddf = dd.from_pandas(df, npartitions=self.config.get('npartitions', 4))
        return ddf
    
    def run(self):
        """
        Executes the ETL process for preparing a tabular dataset.
        This method processes data from either a Snowflake database or local files,
        applies transformations, and saves the final dataset to a specified output location.
        
        Returns:
            str: Path to the output file
        """
        # Determine if we're using Snowflake or local files
        data_source = self.config.get('data_source', 'local')
        
        if data_source.lower() == 'snowflake' and self.snowpark_session:
            # Process data from Snowflake
            logger.info("Processing data from Snowflake via Snowpark")
            ddf = self._process_snowpark_data()
        else:
            # Original local file processing
            logger.info("Processing data from local files")
            df_path = self.config.get('tabular', None)
            npartitions = self.config.get('npartitions', None)
            ddf = dd.read_parquet(df_path, engine='pyarrow')
            
            if npartitions:
                ddf = ddf.repartition(npartitions=npartitions)

        filenames = self.config.keys()

        # Keep only names with normal extensions
        filenames = [f for f in filenames if any([f.endswith(ext) for ext in ['.csv', '.csv.gz', '.parquet', '.feather']])]
        path = self.config['absolute_path']
        filenames = [os.path.join(path, f) for f in filenames]

        # Open empty dask dataframe
        df_pp = None

        logging.info(f"Filenames: {filenames}")
        # Create the requested columns from each filename
        for o, f in zip(filenames, filenames):
            # First check if we have this file checkpointed
            current_name = f.split("/")[-1].split(".")[0]
            saved_loc = f"{self.config['preprocessing']}/{current_name}_preprocessed.parquet"
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
            vals = self.config[o.split("/")[-1]]
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
            aggs = self.config.get('PreTransforms', None)
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
            cols_for_aggregations = list(set(l_cols).intersection(aggs.keys())) if aggs else []

            # Apply aggregations
            if cols_for_aggregations:
                df = transform_aggregations(df, aggs, cols_for_aggregations)

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
        end_transforms = self.config.get('MergedTransforms', None)
        df_pp = merged_transforms(df_pp, end_transforms)
        
        # Reduce to final specified columns
        df_pp = df_pp[self.config["final_cols"]]
        print(df_pp.head(20))
        print(f"Final shape: {df_pp.shape}")
        
        df_pp.to_parquet(self.output_path)
        
        return self.output_path
