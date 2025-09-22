from typing import Any, Union
import pandas as pd
import logging
from matplotlib.pyplot import plt

from .config import *
from . import access

# Set up logging
logger = logging.getLogger(__name__)

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded.
How are missing values encoded, how are outliers encoded? What do columns represent,
makes rure they are correctly labeled. How is the data indexed. Crete visualisation
routines to assess the data (e.g. in bokeh). Ensure that date formats are correct
and correctly timezoned."""


def data() -> Union[pd.DataFrame, Any]:
    """
    Load the data from access and ensure missing values are correctly encoded as well as
    indices correct, column names informative, date and times correctly formatted.
    Return a structured data structure such as a data frame.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR DATA ASSESSMENT CODE:
       - Load data using the access module
       - Check for missing values and handle them appropriately
       - Validate data types and formats
       - Clean and prepare data for analysis

    2. ADD ERROR HANDLING:
       - Handle cases where access.data() returns None
       - Check for data quality issues
       - Validate data structure and content

    3. ADD BASIC LOGGING:
       - Log data quality issues found
       - Log cleaning operations performed
       - Log final data summary

    4. EXAMPLE IMPLEMENTATION:
       df = access.data()
       if df is None:
           print("Error: No data available from access module")
           return None

       print(f"Assessing data quality for {len(df)} rows...")
       # Your data assessment code here
       return df
    """
    logger.info("Starting data assessment")

    # Load data from access module
    df = access.data()

    # Check if data was loaded successfully
    if df is None:
        logger.error("No data available from access module")
        print("Error: Could not load data from access module")
        return None

    logger.info(f"Assessing data quality for {len(df)} rows, {len(df.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your data assessment code here

        # Example: Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found missing values: {missing_counts.to_dict()}")
            print(f"Missing values found: {missing_counts.sum()} total")

        # Example: Check data types
        logger.info(f"Data types: {df.dtypes.to_dict()}")

        # Example: Basic data cleaning (students should customize this)
        # Remove completely empty rows
        df_cleaned = df.dropna(how="all")
        if len(df_cleaned) < len(df):
            logger.info(f"Removed {len(df) - len(df_cleaned)} completely empty rows")

        logger.info(f"Data assessment completed. Final shape: {df_cleaned.shape}")
        return df_cleaned

    except Exception as e:
        logger.error(f"Error during data assessment: {e}")
        print(f"Error assessing data: {e}")
        return None


def query(data: Union[pd.DataFrame, Any]) -> str:
    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data: Union[pd.DataFrame, Any]) -> None:
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError


def labelled(data: Union[pd.DataFrame, Any]) -> Union[pd.DataFrame, Any]:
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes a DataFrame's column names by cleaning and standardizing them.

    This function performs the following steps on the column names:
    1. Removes leading and trailing whitespace.
    2. Converts all characters to lowercase.
    3. Replaces spaces and hyphens with a single underscore.
    4. Removes any characters that are not alphanumeric or underscores.

    Args:
        df (pd.DataFrame): The input pandas DataFrame with columns to normalize.

    Returns:
        pd.DataFrame: The DataFrame with its columns normalized.

    Raises:
        TypeError: If the input is not a pandas DataFrame.
        Exception: Catches any unexpected errors during the column normalization process.

    Example:
        >>> import pandas as pd
        >>> data = {'Col Name 1': [1, 2], ' Another-Col ': [3, 4], 'yet_another--col': [5, 6]}
        >>> df = pd.DataFrame(data)
        >>> normalize_columns(df)
           col_name_1  another_col  yet_another_col
        0           1            3                5
        1           2            4                6
    """
    logger.info("Starting column normalization.")
    try:
        # Check if the input is a DataFrame
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Invalid input type: {type(df)}. Expected pandas.DataFrame.")
            raise TypeError("Input must be a pandas DataFrame.")

        # Original columns
        original_cols = df.columns.tolist()
        logger.debug(f"Original columns: {original_cols}")

        # Normalization
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(
                r"[\s-]+", "_", regex=True
            )  # Replaces one or more spaces or hyphens with a single underscore
            .str.replace(r"[^\w_]", "", regex=True)
        )

        # New columns
        new_cols = df.columns.tolist()
        logger.info(f"Column normalization completed. New columns: {new_cols}")

        return df

    except Exception as e:
        logger.error(f"An unexpected error occurred during column normalization: {e}")
        raise


kenya_counties = [
    "Mombasa",
    "Kwale",
    "Kilifi",
    "Tana River",
    "Lamu",
    "Taita/Taveta",
    "Garissa",
    "Wajir",
    "Mandera",
    "Marsabit",
    "Isiolo",
    "Meru",
    "Tharaka-Nithi",
    "Embu",
    "Kitui",
    "Machakos",
    "Makueni",
    "Nyandarua",
    "Nyeri",
    "Kirinyaga",
    "Murang'a",
    "Kiambu",
    "Turkana",
    "West Pokot",
    "Samburu",
    "Trans Nzoia",
    "Uasin Gishu",
    "Elgeyo/Marakwet",
    "Nandi",
    "Baringo",
    "Laikipia",
    "Nakuru",
    "Narok",
    "Kajiado",
    "Kericho",
    "Bomet",
    "Kakamega",
    "Vihiga",
    "Bungoma",
    "Busia",
    "Siaya",
    "Kisumu",
    "Homa Bay",
    "Migori",
    "Kisii",
    "Nyamira",
    "Nairobi City",
]


def split_region_to_county_subcounty(
    df: pd.DataFrame, region_column: str = "region"
) -> pd.DataFrame:
    """
    Splits a hierarchical 'region' column into 'county' and 'subcounty' columns for Kenya.

    Logic:
    - 'Kenya' represents the country level (county=None, subcounty=None)
    - Any name in the kenya_counties list represents a county (county=name, subcounty=None)
    - All other names are treated as subcounties (county=most recent county, subcounty=name)

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'region' column.
        region_column (str, optional): Name of the column containing hierarchical region
        names. Defaults to 'region'.

    Returns:
        pd.DataFrame: DataFrame with 'county' and 'subcounty' columns, 'region' column dropped.

    Raises:
        TypeError: If the input is not a pandas DataFrame.
        KeyError: If the specified region_column does not exist in the DataFrame.
        Exception: For any unexpected errors.

    Example:
        >>> data = {'region': ['Kenya', 'Nairobi City', 'Westlands']}
        >>> df = pd.DataFrame(data)
        >>> split_region_to_county_subcounty(df)
            county       subcounty
        0     None           None
        1  Nairobi City       None
        2  Nairobi City   Westlands
    """
    logger.info("Starting split of region into county and subcounty.")

    try:
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Invalid input type: {type(df)}. Expected pandas.DataFrame.")
            raise TypeError("Input must be a pandas DataFrame.")

        if region_column not in df.columns:
            logger.error(f"Column '{region_column}' not found in DataFrame.")
            raise KeyError(f"Column '{region_column}' does not exist in DataFrame.")

        df_copy = df.copy()
        df_copy["county"] = None
        df_copy["subcounty"] = None

        current_county = None

        for idx, row in df_copy.iterrows():
            region_name = row[region_column]

            if region_name == "Kenya":
                df_copy.at[idx, "county"] = None
                df_copy.at[idx, "subcounty"] = None
                current_county = None
                logger.debug(f"Row {idx}: Country level detected.")

            elif region_name in kenya_counties:
                df_copy.at[idx, "county"] = region_name
                df_copy.at[idx, "subcounty"] = None
                current_county = region_name
                logger.debug(f"Row {idx}: County detected - {region_name}.")

            else:
                df_copy.at[idx, "county"] = current_county
                df_copy.at[idx, "subcounty"] = region_name
                logger.debug(
                    f"Row {idx}: Subcounty detected - {region_name}, "
                    " assigned to county {current_county}."
                )

        logger.info("Region split completed successfully.")
        return df_copy.drop(columns=[region_column])

    except Exception as e:
        logger.error(f"An error occurred while splitting region: {e}")
        raise

def plot_barh_counts(
    counts_series: pd.Series,
    title: str = 'Bar Chart',
    xlabel: str = 'Count',
    ylabel: str = 'Category',
    color: str = 'skyblue'
) -> None:
    """
    Plots a horizontal bar chart for a pandas Series obtained from value_counts().

    This function generates a horizontal bar chart with the highest values at the top.

    Args:
        counts_series (pd.Series): A pandas Series, typically from value_counts().
        title (str, optional): Chart title. Defaults to 'Bar Chart'.
        xlabel (str, optional): Label for the x-axis. Defaults to 'Count'.
        ylabel (str, optional): Label for the y-axis. Defaults to 'Category'.
        color (str or list, optional): Color of the bars. 
        Can be a single color or a list of colors. Defaults to 'skyblue'.

    Returns:
        None: Displays the plot.

    Raises:
        TypeError: If counts_series is not a pandas Series.
        Exception: Catches any unexpected errors during plotting.

    Example:
        >>> import pandas as pd
        >>> data = ['A', 'B', 'A', 'C', 'B', 'A']
        >>> counts = pd.Series(data).value_counts()
        >>> plot_barh_counts(counts, title="Example Chart", color="lightgreen")
    """
    logger.info("Starting horizontal bar chart plotting.")
    
    try:
        if not isinstance(counts_series, pd.Series):
            logger.error(f"Invalid input type: {type(counts_series)}. Expected pandas.Series.")
            raise TypeError("counts_series must be a pandas Series.")

        logger.debug(f"Counts series:\n{counts_series}")

        plt.figure(figsize=(10, 8))
        counts_series.plot(kind='barh', color=color)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.gca().invert_yaxis()  # highest values at the top
        plt.tight_layout()
        plt.show()

        logger.info("Horizontal bar chart plotted successfully.")

    except Exception as e:
        logger.error(f"An error occurred while plotting bar chart: {e}")
        raise