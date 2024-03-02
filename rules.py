from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Table, MetaData
from flask import current_app
from database_utils import *
from collections import Counter
from datetime import time
import json
import numpy as np
import pandas as pd
import pymysql
import re
from datetime import datetime

pymysql.install_as_MySQLdb()

metadata = MetaData()
# Define your table name as a variable
TABLE_NAME = 'db_messy'



class DataCleaner:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def correct_date_format(self, value):
        """Corrects the date format to YYYY-MM-DD if possible, otherwise returns the original value."""
        try:
            return pd.to_datetime(value, errors='coerce').strftime('%Y-%m-%d')
        except:
            return value

    def capitalize_string_values(self, value):
        """Capitalizes the first letter of each word in a string."""
        if isinstance(value, str):
            return ' '.join(word.capitalize() for word in re.split(' |_|-|!', value))
        else:
            return value

    def replace_null_with_none(self, value):
        """Replaces null or empty cells with 'None'."""
        if pd.isna(value) or value == '':
            return 'None'
        else:
            return value

    def apply_date_format_rule(self):
        """Applies the date format rule to the dataframe."""
        for col in self.dataframe.columns:
            if self.infer_datatype(self.dataframe[col]) == 'date':
                self.dataframe[col] = self.dataframe[col].apply(self.correct_date_format)

    def apply_capitalization_rule(self):
        """Applies the capitalization rule to the dataframe."""
        for col in self.dataframe.columns:
            if self.dataframe[col].dtype == 'O':  # Object type typically indicates string in pandas
                self.dataframe[col] = self.dataframe[col].apply(self.capitalize_string_values)

    def apply_null_replacement_rule(self):
        """Replaces 'None' values with empty string in the dataframe."""
        for col in self.dataframe.columns:
            self.dataframe[col] = self.dataframe[col].apply(lambda x: '' if x == 'None' else x)

    @staticmethod
    def infer_datatype(column):
        # Count occurrences of each data type
        type_counts = Counter()
        for val in column:
            if pd.isna(val):
                continue
            elif isinstance(val, int):
                type_counts['integer'] += 1
            elif isinstance(val, float):
                type_counts['float'] += 1
            elif isinstance(val, str):
                if val.isdigit():
                    type_counts['integer'] += 1
                elif val.replace('.', '', 1).isdigit():
                    type_counts['float'] += 1
                else:
                    try:
                        pd.to_datetime(val)
                        type_counts['date'] += 1
                    except:
                        type_counts['string'] += 1
            else:
                type_counts['other'] += 1

        # Determine the most common data type
        if type_counts:  # Check if type_counts is not empty
            most_common_type, _ = type_counts.most_common(1)[0]
            return most_common_type
        else:
            return 'unknown'  # Return 'unknown' if type_counts is empty

# Modify the FlaggedValues model to include the original_id column
class FlaggedValues(BaseModel):
    __tablename__ = 'flagged_values'
    id = db.Column(db.Integer, primary_key=True)
    original_id = db.Column(db.Integer, nullable=False)  # Add a column to store the original primary key ID
    f_value = db.Column(db.String(255))  # Changed column name to 'f_value'

def get_primary_key(table):
    # Assuming primary key variations
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
            return []
    else:
        # Handle the case where primary_key_col is None
        return []

def apply_rules_to_database(table_name):
    with app.app_context():
        model_class = create_model_class(table_name)
        dataframe = load_data_from_database(model_class)
        cleaned_dataframe, flagged_values, primary_keys, column_datatypes = apply_rules(dataframe, model_class)
        update_database_with_cleaned_data(cleaned_dataframe, model_class)
        save_flagged_values_to_database(flagged_values, primary_keys, cleaned_dataframe, model_class)
    
    print("\nUnfiltered Flagged Values with their Primary Keys and Datatypes:\n")
    for key, values in flagged_values.items():
        if values:
            for i, value in enumerate(values):
                print(f"ID: '{primary_keys[key][i]}' column> {key}: '{value}', Datatype: '{column_datatypes[key]}'")
        
    # Filter and flatten flagged values and primary keys
    filtered_flagged_values = [value for values in flagged_values.values() if values for value in values]
    filtered_primary_keys = [key for keys in primary_keys.values() if keys for key in keys]
    filtered_column_datatypes = {key: column_datatypes[key] for key in flagged_values if flagged_values[key]}

    print("\nFiltered Flagged Values: ", filtered_flagged_values)
    print("\nFiltered Primary Keys: ", filtered_primary_keys)
    print("\nFiltered Column Datatypes: ", filtered_column_datatypes)

    return filtered_flagged_values, filtered_primary_keys, filtered_column_datatypes

def flag_string_values_in_column(column, col_name):
    flagged_values = []
    for value in column:
        if pd.notna(value) and value != '' and not isinstance(value, time) and value != 'None':
            try:
                pd.to_numeric(value, errors='raise', downcast='integer')
            except (ValueError, TypeError):
                flagged_values.append(value)
    return flagged_values

def flag_non_integer_values(column, col_name):
    flagged_values = []
    non_empty_column = column[pd.notna(column) & (column != '') & (column != 'None')]
    numeric_count = non_empty_column.apply(lambda x: pd.to_numeric(x, errors='coerce')).notna().sum()
    if numeric_count / non_empty_column.size > 0.5:
        flagged_values.extend(flag_string_values_in_column(column, col_name))
    return flagged_values

def flag_invalid_dates(column, col_name):
    flagged_values = []
    for value in column:
        if pd.notna(value) and value != '' and value != 'None':
            # Replace spaces with hyphens to standardize the format
            standardized_value = value.replace(' ', '-')
            try:
                # Check if the date is in YYYY-MM-DD format after standardization
                datetime.strptime(standardized_value, '%Y-%m-%d')
            except ValueError:
                # Flag the date if it's not in the correct format
                flagged_values.append(value)
    return flagged_values

def apply_rules(dataframe, model_class):
    cleaner = DataCleaner(dataframe)
    cleaner.apply_capitalization_rule()
    cleaner.apply_null_replacement_rule()

    cleaned_dataframe = dataframe.copy()
    flagged_values = {col: [] for col in cleaned_dataframe.columns}
    primary_keys = {col: [] for col in cleaned_dataframe.columns}
    column_datatypes = {col: cleaner.infer_datatype(dataframe[col]) for col in dataframe.columns}

    for col in cleaned_dataframe.columns:
        if np.issubdtype(cleaned_dataframe[col].dtype, np.number):
            flagged_values[col] = flag_non_integer_values(cleaned_dataframe[col], col)
        else:
            if cleaned_dataframe[col].dtype == np.dtype('O') and not cleaned_dataframe[col].apply(
                    lambda x: isinstance(x, time)).any():
                numeric_values_exist = any(pd.to_numeric(cleaned_dataframe[col], errors='coerce').notna())
                if numeric_values_exist:
                    flagged_values[col] = flag_string_values_in_column(cleaned_dataframe[col], col)

        # Check for date columns, standardize format and flag invalid dates
        if column_datatypes[col] == 'date':
            cleaned_dataframe[col] = cleaned_dataframe[col].apply(lambda x: standardize_date_format(x) if pd.notna(x) else x)
            flagged_values[col].extend(flag_invalid_dates(cleaned_dataframe[col], col))

        # Get primary keys for the flagged values
        primary_key_col = get_primary_key(model_class.__table__)
        if primary_key_col is not None:
            if primary_key_col in cleaned_dataframe.columns:
                primary_keys[col] = get_primary_keys_for_flagged_values(cleaned_dataframe, col, primary_key_col, flagged_values)
            else:
                primary_keys[col] = []
        else:
            primary_keys[col] = []

    return cleaned_dataframe, flagged_values, primary_keys, column_datatypes

def standardize_date_format(date_value):
    """Standardizes date formats by adding hyphens."""
    if isinstance(date_value, str):
        # Replace spaces with hyphens to standardize the format
        return date_value.replace(' ', '-')
    return date_value

# Save flagged values to a database table with primary keys
def save_flagged_values_to_database(flagged_values, primary_keys, cleaned_dataframe, model_class):
    # Create the flagged values table if it doesn't exist
    db.create_all()

    # Iterate through the flagged values and save them to the database with original primary key ID
    with app.app_context():
        for col_name, values in flagged_values.items():
            primary_key_col = get_primary_key(model_class.__table__)
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
        
#To set extend_existing=True
db.Table('flagged_values', metadata, extend_existing=True)

# Create the flagged values table if it doesn't exist
with app.app_context():
    # YourTable is a SQLAlchemy model
    model_class = create_model_class('db_messy')

    # Load data from the database
    dataframe = load_data_from_database(model_class)

    # Define primary_keys here before creating the table
    primary_keys = {col: [] for col in dataframe.columns}

    db.create_all()

    # Apply rules to clean the data and identify flagged values
    cleaned_dataframe, flagged_values, primary_keys, column_datatypes = apply_rules(dataframe, model_class)

    # Iterate through the flagged values and save them to the database with original primary key ID
    for col_name, values in flagged_values.items():
        primary_key_col = get_primary_key(model_class.__table__)  # Define primary_key_col here
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
    
def update_database_with_cleaned_data(cleaned_dataframe, model_class):
    with app.app_context():
        primary_key_col = get_primary_key(model_class.__table__)
        if primary_key_col is None:
            print("Primary key column not found.")
            return

        for index, row in cleaned_dataframe.iterrows():
            # Retrieve the record using the primary key with the new method
            record = db.session.get(model_class, row[primary_key_col])
            if record:
                # Update each column in the record
                for col in cleaned_dataframe.columns:
                    setattr(record, col, row[col])
                db.session.commit()

def MostFrequentFlaggedValue(flagged_values):
    value_counts = Counter()
    for col, values in flagged_values.items():
        for value in values:
            value_counts[value] += 1

    print("\nFlagged Value Counts:")
    for value, count in value_counts.items():
        print(f"{value}: {count}")

    most_common_value, _ = value_counts.most_common(1)[0]
    return most_common_value

most_frequent_flagged_value = MostFrequentFlaggedValue(flagged_values)
print("\nMost Frequent Flagged Value:", most_frequent_flagged_value)

# Assuming primary key variations
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
    apply_rules_to_database(TABLE_NAME)