from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Table, MetaData
from flask import current_app
from database_utils import *
import json
import numpy as np
import datetime
import pandas as pd
import pymysql

pymysql.install_as_MySQLdb()

'''app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/query_data_cleaning'
db = SQLAlchemy(app)'''

metadata = MetaData()


class BaseModel(db.Model):
    __abstract__ = True

# Modify the FlaggedValues model to include the original_id column
class FlaggedValues(BaseModel):
    __tablename__ = 'flagged_values'
    id = db.Column(db.Integer, primary_key=True)
    original_id = db.Column(db.Integer, nullable=False)  # Add a column to store the original primary key ID
    f_value = db.Column(db.String(255))  # Changed column name to 'f_value'

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

def get_primary_key(table):
    # Assuming primary key variations: "ID", "id", "Id"
    primary_key_variations = ["ID", "id", "Id"]

    for col_name in primary_key_variations:
        if col_name in table.columns and table.columns[col_name].primary_key:
            return col_name
    return None

def get_primary_keys_for_flagged_values(dataframe, col, primary_key_col, flagged_values):
    if primary_key_col is not None:
        if primary_key_col in dataframe.columns:
            return dataframe[primary_key_col][dataframe[col].isin(flagged_values[col])].tolist()
        else:
            #print(f"Warning: Primary key column {primary_key_col} not found in DataFrame columns.")
            return []
    else:
        # Handle the case where primary_key_col is None
        return []

def apply_rules_to_database(YourTable):
    with app.app_context():
        dataframe = load_data_from_database()
        cleaned_dataframe, flagged_values = apply_rules(dataframe, YourTable)
        save_flagged_values_to_database(flagged_values, {}, cleaned_dataframe, YourTable)
    print("Changes committed to the database")

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
        flagged_values.extend(flag_string_values_in_column(column, col_name))

    return flagged_values

def apply_rules(dataframe, YourTable):
    cleaned_dataframe = dataframe.copy()
    flagged_values = {col: [] for col in cleaned_dataframe.columns}
    primary_keys = {col: [] for col in cleaned_dataframe.columns}

    for col in cleaned_dataframe.columns:
        if np.issubdtype(cleaned_dataframe[col].dtype, np.number):
            # Process numeric columns
            flagged_values[col] = flag_non_integer_values(cleaned_dataframe[col], col)
        else:
            # Process other columns
            if cleaned_dataframe[col].dtype == np.dtype('O') and not cleaned_dataframe[col].apply(
                    lambda x: isinstance(x, datetime.time)).any():
                # Skip string columns with no numeric values and skip datetime columns
                numeric_values_exist = any(pd.to_numeric(cleaned_dataframe[col], errors='coerce').notna())
                if numeric_values_exist:
                    flagged_values[col] = flag_string_values_in_column(cleaned_dataframe[col], col)

                    # Get primary keys for the flagged values
                    primary_key_col = get_primary_key(YourTable.__table__)
                    if primary_key_col is not None:
                        if primary_key_col in cleaned_dataframe.columns:
                            primary_keys[col] = get_primary_keys_for_flagged_values(cleaned_dataframe, col,
                                                                                   primary_key_col, flagged_values)
                        else:
                            primary_keys[col] = []
                    else:
                        primary_keys[col] = []

    # Save flagged values to a database table with primary keys
    save_flagged_values_to_database(flagged_values, primary_keys, cleaned_dataframe, YourTable)

    # Print flagged values and primary keys
    print("\nFlagged Values with their Primary Keys:")
    for key, values in flagged_values.items():
        if values:
            print(''.join([f"ID: '{primary_keys[key][i]}' column> {key}: '{values[i]}'\n" for i in range(len(values))]))

    return cleaned_dataframe, flagged_values


# Save flagged values to a database table with primary keys
def save_flagged_values_to_database(flagged_values, primary_keys, cleaned_dataframe, YourTable):
    # Create the flagged values table if it doesn't exist
    db.create_all()

    # Iterate through the flagged values and save them to the database with original primary key ID
    with app.app_context():
        for col_name, values in flagged_values.items():
            primary_key_col = get_primary_key(YourTable.__table__)
            primary_keys[col_name] = get_primary_keys_for_flagged_values(cleaned_dataframe, col_name, primary_key_col, flagged_values)
            for flagged_value, original_id in zip(values, primary_keys[col_name]):
                # Check if the flagged value already exists in the database
                existing_entry = FlaggedValues.query.filter_by(original_id=original_id, f_value=flagged_value).first()

                if existing_entry is None:
                    # Flagged value does not exist, add it to the database
                    flagged_value_entry = FlaggedValues(original_id=original_id, f_value=flagged_value)
                    db.session.add(flagged_value_entry)

        # Commit the changes to the database
        db.session.commit()
        print("Changes committed to the database")


# Add the following line to set extend_existing=True
db.Table('flagged_values', metadata, extend_existing=True)

# Create the flagged values table if it doesn't exist
with app.app_context():
    # YourTable is a SQLAlchemy model
    YourTable = create_model_class('db_messy')

    # Load data from the database
    dataframe = load_data_from_database()

    # Define primary_keys here before creating the table
    primary_keys = {col: [] for col in dataframe.columns}

    db.create_all()

    # Apply rules to clean the data and identify flagged values
    cleaned_dataframe, flagged_values = apply_rules(dataframe, YourTable)

    # Iterate through the flagged values and save them to the database with original primary key ID
    for col_name, values in flagged_values.items():
        primary_keys[col_name] = get_primary_keys_for_flagged_values(cleaned_dataframe, col_name, YourTable, flagged_values)
        for flagged_value, original_id in zip(values, primary_keys[col_name]):
            # Check if the flagged value already exists in the database
            existing_entry = FlaggedValues.query.filter_by(original_id=original_id, f_value=flagged_value).first()

            if existing_entry is None:
                # Flagged value does not exist, add it to the database
                flagged_value_entry = FlaggedValues(original_id=original_id, f_value=flagged_value)
                db.session.add(flagged_value_entry)

    # Commit the changes to the database
    db.session.commit()
    print("Changes committed to the database")

def update_database_with_cleaned_data(cleaned_dataframe, YourTable):
    # Do nothing in this function except commit changes
    db.session.commit()

# Assuming primary key variations: "ID", "id", "Id"
primary_key_variations = ["ID", "id", "Id"]

# Load relationships configuration from JSON file
def load_relationships_config():
    with open('relationships_config.json', 'r') as file:
        config = json.load(file)
    return config.get('relationships', [])

# Load relationships configuration from JSON file
relationships_config = load_relationships_config()

# Apply rules to the database during application startup
with app.app_context():
    apply_rules_to_database(YourTable)