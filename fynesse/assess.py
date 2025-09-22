from typing import Any, Union
import pandas as pd
import logging
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns
import numpy as np

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
    title: str = "Bar Chart",
    xlabel: str = "Count",
    ylabel: str = "Category",
    color: str = "skyblue",
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
            logger.error(
                f"Invalid input type: {type(counts_series)}. Expected pandas.Series."
            )
            raise TypeError("counts_series must be a pandas Series.")

        logger.debug(f"Counts series:\n{counts_series}")

        plt.figure(figsize=(10, 8))
        counts_series.plot(kind="barh", color=color)
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


def plot_facilities_distribution(
    shapefile_path: str,
    facilities_df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    county_color: str = "none",
    county_edgecolor: str = "red",
    point_color: str = "green",
    point_size: int = 5,
    point_alpha: float = 0.7,
    title: str = "Health Facilities Distribution by Counties",
    figsize: tuple = (8, 8),
) -> None:
    """
    Plots health facility locations on top of county boundaries from a shapefile.

    Args:
        shapefile_path (str): Path to the shapefile containing county boundaries.
        facilities_df (pd.DataFrame): DataFrame with facility coordinates.
        lat_col (str, optional): Column name for latitude. Defaults to "latitude".
        lon_col (str, optional): Column name for longitude. Defaults to "longitude".
        county_color (str, optional): Fill color for counties. Defaults to "none".
        county_edgecolor (str, optional): Edge color for counties. Defaults to "red".
        point_color (str, optional): Color of facility points. Defaults to "green".
        point_size (int, optional): Marker size for facility points. Defaults to 5.
        point_alpha (float, optional): Transparency for facility points. Defaults to 0.7.
        title (str, optional): Title for the plot. Defaults to
            "Health Facilities Distribution by Counties".
        figsize (tuple, optional): Size of the plot (width, height). Defaults to (8, 8).

    Returns:
        None: Displays the map plot.

    Raises:
        FileNotFoundError: If the shapefile path is invalid.
        KeyError: If latitude/longitude columns are missing in the DataFrame.
        Exception: For any unexpected errors during plotting.

    Example:
        >>> plot_facilities_distribution(
        ...     "County.shp", df_health_facilities,
        ...     lat_col="latitude", lon_col="longitude"
        ... )
    """
    logger.info("Starting facilities distribution plotting.")

    try:
        # Load shapefile
        logger.debug(f"Reading shapefile from {shapefile_path}")
        counties = gpd.read_file(shapefile_path)

        # Check coordinate columns
        if lat_col not in facilities_df.columns or lon_col not in facilities_df.columns:
            logger.error(
                f"Missing required columns: {lat_col} and {lon_col} in facilities_df"
            )
            raise KeyError(f"Missing required columns: {lat_col}, {lon_col}")

        # Convert facilities to GeoDataFrame
        logger.debug("Converting facilities DataFrame to GeoDataFrame.")
        points = gpd.GeoDataFrame(
            facilities_df,
            geometry=[
                Point(xy) for xy in zip(facilities_df[lon_col], facilities_df[lat_col])
            ],
            crs="EPSG:4326",
        )

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        counties.plot(
            ax=ax, color=county_color, edgecolor=county_edgecolor, linewidth=1.5
        )
        points.plot(ax=ax, color=point_color, markersize=point_size, alpha=point_alpha)

        plt.title(title)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.show()

        logger.info("Facilities distribution plotted successfully.")

    except FileNotFoundError as fnf_err:
        logger.error(f"Shapefile not found: {fnf_err}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while plotting facilities distribution: {e}")
        raise


sns.set(style="whitegrid")


def plot_health_facility_analysis(df: pd.DataFrame, plot_type: str = "heatmap") -> None:
    """
    Generate exploratory plots for health facility data.

    Available plot types:
    - "heatmap"       : Correlation heatmap of numeric columns
    - "scatter"       : Facilities vs population density (linear scale)
    - "log_scatter"   : Facilities vs population density (log-log scale)
    - "boxplot"       : Population density distribution by facility level
    - "histogram"     : Facilities per 10,000 population

    Args:
        df (pd.DataFrame): Input dataframe with health facility data.
        plot_type (str, optional): The plot type to generate.
                                   Defaults to "heatmap".

    Returns:
        None

    Raises:
        TypeError: If input is not a pandas DataFrame.
        ValueError: If plot_type is not recognized.

    Example:
        >>> plot_health_facility_analysis(merged_df, plot_type="scatter")
    """
    try:
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Invalid input type: {type(df)}. Expected pandas.DataFrame.")
            raise TypeError("Input must be a pandas DataFrame.")

        logger.info(f"Generating plot: {plot_type}")

        # Ensure facility_count exists
        if "facility_count" not in df.columns and {"county", "subcounty"}.issubset(
            df.columns
        ):
            facility_counts = (
                df.groupby(["county", "subcounty"])
                .size()
                .reset_index(name="facility_count")
            )
            df = df.merge(facility_counts, on=["county", "subcounty"], how="left")

        # Precompute correlations
        preferred = [
            "facility_count",
            "population_x",
            "land_area_sq_km",
            "population_density_no_per_sq_km",
            "male",
            "female",
            "intersex",
            "total",
            "households",
            "average_household_size",
            "facility_level",
        ]
        cols_for_corr = [c for c in preferred if c in df.columns]

        if plot_type == "heatmap":
            corr_df = df[cols_for_corr].corr(method="pearson")
            plt.figure(figsize=(9, 7))
            sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="vlag", center=0)
            plt.title("Correlation matrix — Health Facilities Data")
            plt.tight_layout()
            plt.show()

        elif plot_type == "scatter":
            scatter_df = df.dropna(
                subset=["facility_count", "population_density_no_per_sq_km"]
            )
            plt.figure(figsize=(8, 6))
            sns.regplot(
                data=scatter_df,
                x="population_density_no_per_sq_km",
                y="facility_count",
                scatter_kws={"s": 30, "alpha": 0.6},
                line_kws={"color": "red"},
            )
            plt.xlabel("Population density (per sq. km)")
            plt.ylabel("Number of facilities")
            plt.title("Facilities vs Population Density (linear)")
            plt.tight_layout()
            plt.show()

        elif plot_type == "log_scatter":
            scatter_df = df.dropna(
                subset=["facility_count", "population_density_no_per_sq_km"]
            ).copy()
            scatter_df["log_pop_density"] = np.log1p(
                scatter_df["population_density_no_per_sq_km"]
            )
            scatter_df["log_facility_count"] = np.log1p(scatter_df["facility_count"])
            plt.figure(figsize=(8, 6))
            sns.regplot(
                data=scatter_df,
                x="log_pop_density",
                y="log_facility_count",
                scatter_kws={"s": 30, "alpha": 0.6},
                line_kws={"color": "red"},
            )
            plt.xlabel("log(1 + population density)")
            plt.ylabel("log(1 + facility count)")
            plt.title("Facilities vs Population Density (log-log)")
            plt.tight_layout()
            plt.show()

        elif plot_type == "boxplot":
            if "facility_level" not in df.columns:
                logger.error("facility_level column not found in dataframe.")
                raise KeyError("facility_level column required for boxplot.")
            box_df = df.dropna(
                subset=["facility_level", "population_density_no_per_sq_km"]
            )
            plt.figure(figsize=(10, 6))
            sns.boxplot(
                data=box_df, x="facility_level", y="population_density_no_per_sq_km"
            )
            plt.xlabel("Facility level")
            plt.ylabel("Population density (per sq. km)")
            plt.title("Population density distribution by facility level")
            plt.tight_layout()
            plt.show()

        elif plot_type == "histogram":
            if "facility_count" in df.columns and "population_x" in df.columns:
                df["facilities_per_10k"] = (
                    df["facility_count"] / df["population_x"] * 10000
                )
                plt.figure(figsize=(8, 6))
                sns.histplot(df["facilities_per_10k"].dropna(), bins=40, kde=True)
                plt.xlabel("Facilities per 10,000 population")
                plt.title("Distribution of facilities per 10k population")
                plt.tight_layout()
                plt.show()
            else:
                logger.error("facility_count and population_x required for histogram.")
                raise KeyError("facility_count and population_x columns required.")

        else:
            logger.error(f"Unknown plot_type: {plot_type}")
            raise ValueError(
                f"Invalid plot_type '{plot_type}'. "
                f"Choose from ['heatmap','scatter','log_scatter','boxplot','histogram']."
            )

        logger.info(f"Plot {plot_type} generated successfully.")

    except Exception as e:
        logger.error(f"An error occurred while generating plot: {e}")
        raise
