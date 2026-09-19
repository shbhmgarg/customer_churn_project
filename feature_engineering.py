import pandas as pd
import numpy as np


def add_engineered_features(data):
    """Adds all 4 engineered features. Works on any dataframe with the required raw columns.
    This lives in its own module (not a notebook cell) so it can be imported identically
    by both the training notebook and the deployed API — this is required for a pickled
    Pipeline containing a FunctionTransformer to load correctly outside the notebook."""
    data = data.copy()

    # 1. Tenure Group
    data['TenureGroup'] = pd.cut(
        data['tenure'], bins=[-1, 12, 24, 48, 100],
        labels=['0-12 months (New)', '13-24 months (Established)',
                '25-48 months (Loyal)', '48+ months (Very Loyal)']
    )

    # 2. Average Monthly Spend
    data['AvgMonthlySpend'] = np.where(
        data['tenure'] == 0, data['MonthlyCharges'], data['TotalCharges'] / data['tenure']
    )

    # 3. Number of Additional Services
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                     'TechSupport', 'StreamingTV', 'StreamingMovies']
    data['NumAdditionalServices'] = (data[service_cols] == 'Yes').sum(axis=1)

    # 4. High-Risk Profile
    data['IsHighRiskProfile'] = (
        (data['Contract'] == 'Month-to-month') & (data['PaymentMethod'] == 'Electronic check')
    ).astype(int)

    return data