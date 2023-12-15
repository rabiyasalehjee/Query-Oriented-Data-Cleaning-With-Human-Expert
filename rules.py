import pandas as pd
import pymysql
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Table, MetaData
import json
import numpy as np
import datetime

pymysql.install_as_MySQLdb()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/query_data_cleaning'
db = SQLAlchemy(app)

metadata = MetaData()

class BaseModel(db.Model):
    __abstract__ = True

def apply_rules_to_database():
    global YourTable
    # YourTable is a SQLAlchemy model
    YourTable = create_model_class('db_messy')

    # Load data from the database
    dataframe = load_data_from_database(YourTable)

    # Apply rules to clean the data and identify flagged values
    _, flagged_values,  = apply_rules(dataframe)

def load_data_from_database(YourTable):
    # YourTable is a SQLAlchemy model
    data = YourTable.query.all()

    # Extract relevant columns from the query result
    columns_to_include = [column.key for column in YourTable.__table__.columns]
    data_dict_list = [{col: getattr(row, col) for col in columns_to_include} for row in data]

    # Convert the list of dictionaries to a DataFrame
    dataframe = pd.DataFrame(data_dict_list)
    
    return dataframe

def flag_string_values_in_column(column, col_name):
    flagged_values = []

    for value in column:
        if pd.notna(value) and value != '' and not isinstance(value, datetime.time):
            # Skip if value is NaN, None, or empty string, or any datetime values
            try:
                # Try to convert each non-null, non-empty value to an integer
                pd.to_numeric(value, errors='raise', downcast='integer')
            except (ValueError, TypeError):
                # If conversion fails, the value is not an integer and should be flagged
                flagged_values.append(value)

    return flagged_values

def flag_non_integer_values(column, col_name):
    flagged_values = []

    # Exclude empty, null, None, or NaN values from counting
    non_empty_column = column[pd.notna(column) & (column != '')]

    # Count the number of non-null numeric values
    numeric_count = non_empty_column.apply(lambda x: pd.to_numeric(x, errors='coerce')).notna().sum()

    # If the majority of values are numeric, treat the column as an integer column
    if numeric_count / non_empty_column.size > 0.5:
        #print(f"{col_name} is treated as an integer column.")
        flagged_values.extend(flag_string_values_in_column(column, col_name))

    return flagged_values

def apply_rules(dataframe):
    cleaned_dataframe = dataframe.copy()
    flagged_values = {col: [] for col in cleaned_dataframe.columns}

    for col in cleaned_dataframe.columns:

        if np.issubdtype(cleaned_dataframe[col].dtype, np.number):
            # Process numeric columns
            flagged_values[col] = flag_non_integer_values(cleaned_dataframe[col], col)
        else:
            # Process other columns
            if cleaned_dataframe[col].dtype == np.dtype('O') and not cleaned_dataframe[col].apply(lambda x: isinstance(x, datetime.time)).any():
                # Skip string columns with no numeric values and skip datetime columns
                numeric_values_exist = any(pd.to_numeric(cleaned_dataframe[col], errors='coerce').notna())
                if numeric_values_exist:
                    flagged_values[col] = flag_string_values_in_column(cleaned_dataframe[col], col)
    # Print flagged values 
    print("Flagged Values:")
    for key, values in flagged_values.items():
        if values:
            print(f"{key}: {values}")

    return cleaned_dataframe, flagged_values

def update_database_with_cleaned_data(cleaned_dataframe, YourTable):
    # Do nothing in this function except commit changes
    db.session.commit()

def create_model_class(table_name):
    global YourTable  # Declare YourTable as a global variable
    with app.app_context():  # Create tables within the application context
        YourTable = create_model_class_internal(table_name)
    return YourTable

def create_model_class_internal(table_name):
    table = Table(table_name, metadata, autoload_with=db.engine)

    class_name = table_name.capitalize() + 'Model'

    # Check if the class is already defined
    if class_name in globals():
        return globals()[class_name]

    class DynamicModel(BaseModel):
        __table__ = table

    # Set the class name in the global namespace
    globals()[class_name] = DynamicModel

    return DynamicModel

# Assuming primary key variations: "ID", "id", "Id"
primary_key_variations = ["ID", "id", "Id"]

# Load relationships configuration from JSON file
def load_relationships_config():
    with open('relationships_config.json', 'r') as file:
        config = json.load(file)
    return config.get('relationships', [])

# YourTable is a SQLAlchemy model
YourTable = create_model_class('db_messy')

# Load relationships configuration from JSON file
relationships_config = load_relationships_config()

# Apply rules to the database during application startup
with app.app_context():
    apply_rules_to_database()
