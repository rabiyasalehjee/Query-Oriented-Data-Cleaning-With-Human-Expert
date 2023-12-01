import pandas as pd
import re

def apply_rules(dataframe):
    print("Applying Rule 1: Empty cells or cells containing Nan, NaN, Null should be considered as empty cells")
    dataframe.fillna('', inplace=True)

    print("Applying Rule 2: Convert text to lowercase for uniformity")
    dataframe = dataframe.applymap(lambda x: str(x).lower())

    print("Applying Rule 3: Standardize date formats in columns containing the word 'date'")
    date_columns = [col for col in dataframe.columns if re.search(r'date', col, flags=re.IGNORECASE)]
    for col in date_columns:
        dataframe[col] = dataframe[col].apply(lambda x: str(x).lower())

    print("Applying Rule 4: Detect and apply rule for numerical/integral columns")
    for col in dataframe.columns:
        # Use regex to detect numerical/integral columns
        if dataframe[col].apply(lambda x: bool(re.match(r'^[+-]?\d+$', str(x)))).all():
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
            print(f"Applied numerical/integral rule to column: {col}")

    print("Rules applied.")
    return dataframe
