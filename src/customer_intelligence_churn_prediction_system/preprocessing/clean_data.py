from logging import error
import pandas as pd

def clean_customer_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Telco customer churn dataset.

    Args:
        df: Raw customer DataFrame.

    Returns:
        A cleaned copy of the DataFrame.
    """
    if not isinstance(df,pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input dataset is empty.")
    
    cleaned_df = df.copy()
    
    cleaned_df.columns = cleaned_df.columns.str.strip()

    #  Remove unnecessary whitespace from text columns
    text_columns = cleaned_df.select_dtypes(
        include=["object"]
    ).columns

    for col in text_columns:
        cleaned_df[col] = cleaned_df[col].str.strip()

    #  Convert TotalCharges into numeric
    if "TotalCharges" not in cleaned_df.columns:
        raise ValueError("Required column 'TotalCharges' is missing.")
    
    cleaned_df["TotalCharges"] = pd.to_numeric(
        cleaned_df["TotalCharges"],
        errors = "coerce"
    )
    #  Check duplicate customer IDs
    if "customerID" in cleaned_df:
        duplicate_count = cleaned_df["customerID"].duplicated().sum()
        if duplicate_count>0:
            raise ValueError(f"Found {duplicate_count} duplicate customer IDs. "
            "Investigate before removing records.")
    
    cleaned_df.reset_index(drop=True,inplace = True)

    zero_tenure_mask = (
        (cleaned_df["tenure"] == 0)&
        (cleaned_df["TotalCharges"].isnull())
    )

    cleaned_df.loc[zero_tenure_mask,"TotalCharges"] = 0

    return cleaned_df
    

