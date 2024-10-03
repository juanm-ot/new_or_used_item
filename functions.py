import logging
import numpy as np

def create_boolean_columns(dtf, columns):
    """
    Automates the creation of boolean columns indicating whether the specified columns
    contain data (not null). For each column, a new boolean column named 'have_{column_name}'
    is created, where 1 indicates the presence of data and 0 indicates missing data.

    Parameters:
    dtf (pd.DataFrame): The DataFrame to be processed.
    columns (list): List of column names to process.

    Returns:
    pd.DataFrame: The modified DataFrame with the new boolean columns.

    Example:
    columns_to_process = ['official_store_id', 'parent_item_id', 'video_id', 'original_price', 'seller_contact']
    dtf = create_boolean_columns(dtf, columns_to_process)
    """
    try:
        logging.info(f"Creating boolean columns: {columns}")
        for col in columns:
            if col in dtf.columns:
                # Create a boolean column indicating if the original column has data
                dtf[f'have_{col}'] = dtf[col].notnull().astype(int)
            else:
                logging.warning(f"Warning: Column '{col}' not found in DataFrame.")
        logging.info("Boolean columns created successfully.")
        return dtf
    except Exception as e:
        logging.error(f"Error in create_boolean_columns: {e}")
        raise

def process_columns_with_empty_lists(dtf, columns):
    """
    Processes the specified columns in the DataFrame by converting them to strings, 
    replacing empty lists ('[]') with NaN, and creating a new boolean column indicating 
    whether the original column contains data or is missing.

    Parameters:
    dtf (pd.DataFrame): The DataFrame to be processed.
    columns (list): List of column names to be processed.

    Returns:
    pd.DataFrame: The modified DataFrame with additional boolean columns 
                  in the format 'have_{column_name}' for each processed column.

    Behavior:
    - Converts each specified column to string format.
    - Replaces any occurrence of '[]' (empty list representation) with NaN.
    - Creates a new column 'have_{column_name}' with a boolean indicator: 
      1 if the original column has data, 0 otherwise.
    - If a column is not found in the DataFrame, it prints a warning.
    
    """

    try:
        logging.info(f"Processing columns with empty lists: {columns}")
        for col in columns:
            if col in dtf.columns:
                # Convert the column to string and replace empty lists with NaN
                dtf[col] = dtf[col].astype(str)
                dtf[col] = dtf[col].replace('[]', np.nan)
                # Create a boolean column indicating if there is data
                dtf[f'have_{col}'] = dtf[col].notnull().astype(int)
            else:
                logging.warning(f"Warning: Column '{col}' not found in DataFrame.")
        logging.info("Columns processed successfully.")
        return dtf
    except Exception as e:
        logging.error(f"Error in process_columns_with_empty_lists: {e}")
        raise