import pandas as pd
import numpy as np

def build_customer_features(df:pd.DataFrame) -> pd.DataFrame:
    """
    Create business-oriented features for customer churn prediction.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input dataset is empty.")

    featured_df = df.copy()

    #  Tenure groups
    featured_df["TenureGroup"] = pd.cut(
        featured_df["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["New", "Short-Term", "Medium-Term", "Long-Term"]
    )

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]
    available_services = featured_df[service_columns].eq("Yes")
    featured_df["ServiceCount"] = available_services.sum(axis=1)

    #  Month-to-month contract indicator
    featured_df["IsMonthToMonth"] = (
        featured_df["Contract"] == "Month-to-month"
    ).astype(int)

    #  Average monthly spending
    featured_df["AverageMonthlySpend"] = np.where(
        featured_df["tenure"] > 0,
        featured_df["TotalCharges"] / featured_df["tenure"],
        0
    )

    return featured_df