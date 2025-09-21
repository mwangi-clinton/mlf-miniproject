"""
Access module for the fynesse framework.

This module handles data access functionality including:
- Data loading from various sources (web, local files, databases)
- Legal compliance (intellectual property, privacy rights)
- Ethical considerations for data usage
- Error handling for access issues

Legal and ethical considerations are paramount in data access.
Ensure compliance with e.g. .GDPR, intellectual property laws, and ethical guidelines.

Best Practice on Implementation
===============================

1. BASIC ERROR HANDLING:
   - Use try/except blocks to catch common errors
   - Provide helpful error messages for debugging
   - Log important events for troubleshooting

2. WHERE TO ADD ERROR HANDLING:
   - File not found errors when loading data
   - Network errors when downloading from web
   - Permission errors when accessing files
   - Data format errors when parsing files

3. SIMPLE LOGGING:
   - Use print() statements for basic logging
   - Log when operations start and complete
   - Log errors with context information
   - Log data summary information

4. EXAMPLE PATTERNS:
   
   Basic error handling:
   try:
       df = pd.read_csv('data.csv')
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
   
   With logging:
   print("Loading data from data.csv...")
   try:
       df = pd.read_csv('data.csv')
       print(f"Successfully loaded {len(df)} rows of data")
       return df
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
"""

from typing import Literal, TypedDict, List, Union

import pandas as pd

import os
import re
import csv
import json
from pypdf import PdfReader
import logging

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def data() -> Union[pd.DataFrame, None]:
    """
    Read the data from the web or local file, returning structured format such as a data frame.

    IMPLEMENTATION GUIDE
    ====================

    1. REPLACE THIS FUNCTION WITH YOUR ACTUAL DATA LOADING CODE:
       - Load data from your specific sources
       - Handle common errors (file not found, network issues)
       - Validate that data loaded correctly
       - Return the data in a useful format

    2. ADD ERROR HANDLING:
       - Use try/except blocks for file operations
       - Check if data is empty or corrupted
       - Provide helpful error messages

    3. ADD BASIC LOGGING:
       - Log when you start loading data
       - Log success with data summary
       - Log errors with context

    4. EXAMPLE IMPLEMENTATION:
       try:
           print("Loading data from data.csv...")
           df = pd.read_csv('data.csv')
           print(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
           return df
       except FileNotFoundError:
           print("Error: data.csv file not found")
           return None
       except Exception as e:
           print(f"Error loading data: {e}")
           return None

    Returns:
        DataFrame or other structured data format
    """
    logger.info("Starting data access operation")

    try:
        # IMPLEMENTATION: Replace this with your actual data loading code
        # Example: Load data from a CSV file
        logger.info("Loading data from data.csv")
        df = pd.read_csv("data.csv")

        # Basic validation
        if df.empty:
            logger.warning("Loaded data is empty")
            return None

        logger.info(
            f"Successfully loaded data: {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    except FileNotFoundError:
        logger.error("Data file not found: data.csv")
        print("Error: Could not find data.csv file. Please check the file path.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        print(f"Error loading data: {e}")
        return None


class TableConfig(TypedDict):
    headers: List[str]
    expected_columns: int


def extract_census_data(
    pdf_file_path: str,
    output_format: Literal["csv", "xls", "xlsx", "json"],
    data_title: str,
    output_file: str,
) -> None:
    """
    Extract 2019 Kenya census data from a PDF and save it in a specified format.

    Args:
        pdf_file_path (str): Path to the PDF file to extract data from.
        output_format (Literal["csv", "xls", "xlsx", "json"]): Format to save
            the extracted data.
        data_title (str): Specific title of the data table to extract.
        output_file (str, optional): Base name for the output file. Defaults to "output".

    Returns:
        None: The extracted data is saved directly to disk in the specified format.

    Raises:
        FileNotFoundError: If the PDF file cannot be found at the given path.
        ValueError: If `output_format` is not one of the supported formats.
        RuntimeError: If extraction from the PDF fails unexpectedly.

    Example:
        >>> extract_census_data("census2019.pdf", "csv",
          "Distribution of Population by Sex and Sub-County,output_file")
    """

    logger.info(
        f"Starting extraction for data_title: '{data_title}' "
        f"from PDF: '{pdf_file_path}' "
        f"with output_format: '{output_format}' "
        f"and output_file: '{output_file}'"
    )

    if not os.path.exists(pdf_file_path):
        logger.error(f"PDF file not found: {pdf_file_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_file_path}")

    valid_formats = {"csv", "xls", "xlsx", "json"}
    if output_format not in valid_formats:
        logger.error(
            f"Unsupported format: {output_format}. Must be one of {valid_formats}"
        )
        raise ValueError(
            f"Unsupported format: {output_format}. Must be one of {valid_formats}"
        )

    # Table configurations
    table_configs: dict[str, TableConfig] = {
        (
            "Distribution of Population, Number of Households and Average "
            "Household Size by Sub-County"
        ): {
            "headers": [
                "Region",
                "Population",
                "Households",
                "Average Household Size",
            ],
            "expected_columns": 3,
        },
        "Distribution of Population by Sex and Sub-County": {
            "headers": ["Region", "Male", "Female", "Intersex", "Total"],
            "expected_columns": 4,
        },
        "Distribution of Population by Land Area and Population Density by Sub-County": {
            "headers": [
                "Region",
                "Land Area (Sq. Km)",
                "Population",
                "Population Density (No. per Sq. Km)",
            ],
            "expected_columns": 3,
        },
    }

    if data_title not in table_configs:
        logger.error(
            f"Unsupported data title: {data_title}. Must be one of {list(table_configs.keys())}"
        )
        raise ValueError(
            f"Unsupported data title: {data_title}. Must be one of {list(table_configs.keys())}"
        )

    config = table_configs[data_title]
    collected_data: List[List] = []  # List of lists containing strings and floats
    skip_phrases = ["table of contents", "list of figures", "list of tables"]

    logger.info(f"Using table config: {config}")

    try:
        logger.info("Opening PDF file...")
        pdf_reader = PdfReader(pdf_file_path)
        logger.info("PDF opened successfully")

        table_found = False
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_content = page.extract_text().lower()

            # Skip unwanted pages
            if any(phrase in page_content for phrase in skip_phrases):
                logger.debug(f"Skipping page {page_num} due to skip phrases")
                continue

            # Search for the data title
            words = re.findall(r"\w+", data_title.lower())
            pattern = r"\b" + r"[\s\S]*?".join(map(re.escape, words)) + r"\b"
            match = re.search(pattern, page_content.lower())

            if match:
                logger.info(f"Table title matched on page {page_num}")
                table_found = True
                start_index = match.end()
                raw_data = page_content[start_index:]

                # Clean the data
                raw_data = re.sub(r"\s+\.\.\s+", " PLACEHOLDER ", raw_data)
                cleaned_data = re.sub(r"[.…]{2,}", " ", raw_data)
                lines = [
                    line.strip() for line in cleaned_data.split("\n") if line.strip()
                ]

                for line in lines:
                    tokens = [tok for tok in line.split() if tok.strip()]

                    if not tokens:
                        continue

                    # Skip header lines
                    line_lower = line.lower()
                    if any(
                        phrase in line_lower
                        for phrase in [
                            "county/sub county",
                            "national/county",
                            "population",
                            "land area",
                            "population density",
                            "sq. km",
                            "no. per sq. km",
                            "cont'd",
                            "intersex",
                            "sex ratio",
                        ]
                    ):
                        continue

                    # Process tokens
                    name_parts: List[str] = []
                    numbers: List[float] = []

                    for tok in tokens:
                        if tok == "PLACEHOLDER" or tok in {".", "..", "..."}:
                            numbers.append(0.0)
                        elif re.match(r"^(\d{1,3}(,\d{3})*(\.\d+)?)$", tok):
                            num = float(
                                tok.replace(",", "")
                            )  # Convert all numbers to float
                            numbers.append(num)
                        else:
                            name_parts.append(tok)

                    name = " ".join(name_parts).title().strip()

                    # Special handling for different table types
                    if "ratio" in name.lower():
                        continue

                    if (
                        data_title == "Distribution of Population by Land Area and "
                        "Population Density by Sub-County"
                        and len(numbers) > 3
                    ):
                        numbers = numbers[-3:]

                    if name and len(numbers) == config["expected_columns"]:
                        collected_data.append([name] + numbers)

        if not table_found:
            logger.warning("Table title not found in the PDF")

        logger.info(
            f"Extraction completed. Collected {len(collected_data)} rows of data"
        )

        if not collected_data:
            logger.warning(
                "No data found with the specified title. Check the title again."
            )
            print("No data found with the specified title. Check the title again.")
            return None

        # Save data to file
        output_path = f"{output_file}.{output_format}"
        logger.info(f"Saving data to {output_path} in {output_format} format")

        if output_format == "csv":
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(config["headers"])  # Type is List[str] from TypedDict
                writer.writerows(collected_data)

        elif output_format in {"xls", "xlsx"}:
            df = pd.DataFrame(collected_data, columns=config["headers"])
            df.to_excel(output_path, index=False)

        elif output_format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"columns": config["headers"], "data": collected_data},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        logger.info(f"Data saved successfully to {output_path}")
        print(f"Data saved successfully to {output_path}")

    except Exception as e:
        logger.error(f"An error occurred while extracting data: {e}")
        print(f"An error occurred while extracting data: {e}")
        raise RuntimeError(f"An error occurred while extracting data: {e}")
