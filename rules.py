import pandas as pd
import pymysql
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Table, MetaData
import json
import numpy as np

pymysql.install_as_MySQLdb()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/query_data_cleaning'
db = SQLAlchemy(app)

metadata = MetaData()
class BaseModel(db.Model):
    __abstract__ = True

def apply_rules_to_database():
    # Assuming YourTable is a SQLAlchemy model
    YourTable = create_model_class('db_messy')

    # Load data from the database
    dataframe = load_data_from_database(YourTable)

    # Apply rules to clean the data and identify flagged values
    cleaned_dataframe, flagged_values = apply_rules(dataframe)
    print("Cleaned DataFrame:")
    print(cleaned_dataframe)
    print("\nFlagged Values:")
    print(flagged_values)

    # Update the database with the cleaned data
    update_database_with_cleaned_data(cleaned_dataframe, YourTable)

    print("Data cleaning and update complete.")

def load_data_from_database(YourTable):
    # Assuming YourTable is a SQLAlchemy model
    data = YourTable.query.all()

    # Extract relevant columns from the query result
    columns_to_include = [column.key for column in YourTable.__table__.columns]
    data_dict_list = [{col: getattr(row, col) for col in columns_to_include} for row in data]

    # Convert the list of dictionaries to a DataFrame
    dataframe = pd.DataFrame(data_dict_list)

    return dataframe

def clean_string_column(column):
    cleaned_column = column.copy()
    flagged_values = []

    # Example: Identify and flag specific string values
    flagged_mask = column.astype(str).str.lower() == 'two'
    flagged_values.extend(cleaned_column[flagged_mask].tolist())
    cleaned_column[flagged_mask] = np.nan

    return cleaned_column, flagged_values

def clean_numeric_column(column):
    cleaned_column = column.copy()
    flagged_values = []

    # Example: Convert non-integer values to NaN and flag them
    non_integer_mask = ~column.astype(str).str.isdigit()
    flagged_values.extend(cleaned_column[non_integer_mask].tolist())
    cleaned_column[non_integer_mask] = np.nan

    return cleaned_column, flagged_values

def apply_rules(dataframe):
    cleaned_dataframe = dataframe.copy()
    flagged_values = {}

    for col in cleaned_dataframe.columns:
        print(f"Processing column: {col}")
        print("Original values:")
        print(cleaned_dataframe[col])

        # Process numeric and string columns
        if np.issubdtype(cleaned_dataframe[col].dtype, np.number):
            print("Processing numeric column.")
            cleaned_dataframe[col], flagged_values[col] = clean_numeric_column(cleaned_dataframe[col])
        else:
            print("Processing string column.")
            cleaned_dataframe[col], flagged_values[col] = clean_string_column(cleaned_dataframe[col])

        # Display flagged values
        if flagged_values[col]:
            print(f"Flagged values in {col}: {flagged_values[col]}")

        # Display cleaned values
        print(f"Cleaned values in {col}:\n{cleaned_dataframe[col]}")

    return cleaned_dataframe, flagged_values


def update_database_with_cleaned_data(cleaned_dataframe, YourTable):
    primary_key_column = 'ID'

    for _, row in cleaned_dataframe.iterrows():
        # Replace NaN values with None or a suitable default value
        row = row.replace({np.nan: None})

        # Update the database row
        db.session.query(YourTable).filter(getattr(YourTable, primary_key_column) == row[primary_key_column]).update(row.to_dict())

    db.session.commit()

def create_model_class(table_name):
    global YourTable  # Declare YourTable as a global variable
    with app.app_context():  # Create tables within the application context
        YourTable = create_model_class_internal(table_name)
    return YourTable

def create_model_class_internal(table_name):
    table = Table(table_name, metadata, autoload_with=db.engine)

    class DynamicModel(BaseModel):
        __table__ = table

    return DynamicModel

# Assuming primary key variations: "ID", "id", "Id"
primary_key_variations = ["ID", "id", "Id"]

# Load relationships configuration from JSON file
def load_relationships_config():
    with open('relationships_config.json', 'r') as file:
        config = json.load(file)
    return config.get('relationships', [])

# Assuming YourTable is a SQLAlchemy model
YourTable = create_model_class('db_messy')

# Load relationships configuration from JSON file
relationships_config = load_relationships_config()

# Apply rules to the database during application startup
with app.app_context():
    apply_rules_to_database()
