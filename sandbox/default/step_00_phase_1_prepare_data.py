import pandas as pd

def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Loads a dataset from a CSV file into a Pandas DataFrame.

    Parameters:
    csv_path (str): The file path to the CSV dataset.

    Returns:
    pd.DataFrame: The dataset loaded into a DataFrame.
    """
    try:
        # Load the CSV file into a DataFrame
        dataframe = pd.read_csv(csv_path)
        return dataframe
    except FileNotFoundError:
        raise FileNotFoundError(f"The specified file at {csv_path} was not found.")
    except pd.errors.EmptyDataError:
        raise ValueError("The provided CSV file is empty.")
    except pd.errors.ParserError:
        raise ValueError("Error parsing the CSV file.")