from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy import inspect
from sqlalchemy import Table, MetaData
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sqlalchemy.ext.declarative import declarative_base
from database_utils import *
import pandas as pd
import pymysql
import json
import random
import requests
from datetime import datetime
from rules import apply_rules, apply_rules_to_database
from flask_cors import CORS
from relationships import define_relationships
from sqlalchemy import inspect
import logging
from dateutil.parser import parse


# Setup basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


pymysql.install_as_MySQLdb()

YourTable = create_model_class('db_messy')
define_relationships(config_file_path='relationships_config.json')

with app.app_context():  # Create tables within the application context
    YourTable.__table__.create(bind=db.engine, checkfirst=True)

# Load relationships configuration from JSON file
def load_relationships_config():
    with open('relationships_config.json', 'r') as file:
        config = json.load(file)
    return config.get('relationships', [])

def fetch_data_with_primary_key():
    # Fetch data including primary key
    data_with_primary_key = YourTable.query.all()
    return data_with_primary_key

# Find related column based on relationships configuration
def find_related_column(column, relationships):
    for relationship in relationships:
        if column == relationship['main_column']:
            return relationship['related_column']
    return None

def update_database_with_cleaned_data(cleaned_dataframe, YourTable, primary_key_variations):
    try:
        for index, row in cleaned_dataframe.iterrows():
            # Find the primary key column variation that exists in the DataFrame
            primary_key_column = next((col for col in primary_key_variations if col in row.index), None)

            if primary_key_column:
                # Update the row in the database based on the found primary key column
                primary_key_value = row[primary_key_column]
                print(f"Updating row with primary key {primary_key_column}={primary_key_value} in the database")
                db.session.query(YourTable).filter(getattr(YourTable, primary_key_column) == primary_key_value).update(row.to_dict())

        
        db.session.commit()
        #print("Changes committed to the database")
    except Exception as e:
        print(f"Error committing changes to the database: {e}")

# Function to check correction needed
def check_correction_needed(row):
    """
    Check if correction is needed for a given row.
    """
    # Example: Check if any value in the row is missing
    if any(pd.isnull(row)):
        return 1  # Correction needed
    else:
        return 0  # No correction needed
    
# Function to check existing data against rules
def check_existing_data(dataframe):
    existing_data_errors = []

    # Example: Check 'Year' column for non-integer values
    for index, row in dataframe.iterrows():
        try:
            int(row['Year'])
        except ValueError:
            existing_data_errors.append(
                f"Existing data error: The 'Year' data for the year {row['Year']} is not an integer."
            )

    # Add more checks for other columns later

    return existing_data_errors

# Function to train a classification model
def train_classification_model(dataframe, relationships_config):
    # Assume 'Correction_Needed' is a new column indicating whether correction is needed (1) or not (0)
    dataframe['Correction_Needed'] = dataframe.apply(lambda row: check_correction_needed(row), axis=1)

    # Feature extraction
    features = dataframe.apply(lambda row: ' '.join([str(row[column]) for column in dataframe.columns]), axis=1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, dataframe['Correction_Needed'], test_size=0.2, random_state=42)

    # Use TF-IDF Vectorizer to convert text data into numerical features
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train a Random Forest Classifier
    classifier = RandomForestClassifier()
    classifier.fit(X_train_vec, y_train)

    # Evaluate the model on the test set
    y_pred = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

    return classifier, vectorizer

# Function to generate questions using the machine learning model
def generate_questions_ml(dataframe, relationships_config, YourTable):
    # Apply rules
    cleaned_dataframe, flagged_values, primary_keys, column_datatypes  = apply_rules(dataframe, YourTable)# Unpack the tuple

    # Train the classification model
    classifier, vectorizer = train_classification_model(cleaned_dataframe, relationships_config)

    questions = []
    question_count = 0
    # Keep track of rows to avoid duplicate questions
    processed_rows = set()

    # Generate Questions for User
    for index, row in cleaned_dataframe.iterrows():
        # Dynamically fetch column names from the database
        text_columns = [column.name for column in inspect(YourTable).columns]

        feature_vector = vectorizer.transform([str(row[column]) for column in text_columns])

        # Predict using the trained model
        prediction = classifier.predict(feature_vector)[0]

        if index in processed_rows:
            continue

        # Extract the primary key (ID) column name and value
        primary_key_column = 'ID' 
        row_id = row[primary_key_column]

        # Generate a question for each relationship
        for relationship in relationships_config:
            main_column = relationship['main_column']
            related_column = relationship['related_column']
            main_value = row[main_column]
            related_value = row[related_column]
            question_name = f'q_{index}_ml_correct_{main_column}_{related_column}'
            main_column_datatype = determine_column_datatype(dataframe[main_column])  # Determine datatype for main_column
            related_column_datatype = determine_column_datatype(dataframe[related_column])  # Determine datatype for related_column
            
            # Add the row ID to the question data
            question_data = {
                'row_id': row_id,  
                'mainColumn': main_column,
                'mainValue': main_value,
                'relatedColumn': related_column,
                'relatedValue': related_value,
                'mainColumnDatatype': main_column_datatype,
                'relatedColumnDatatype': related_column_datatype,
            }
            #print(f"Question Data (Before Adding to Questions): {question_data}")

            main_value = row[main_column]
            related_value = row[related_column]

            # Check if either main or related value is missing or empty
            if pd.isnull(main_value) or main_value == '' or pd.isnull(related_value) or related_value == '':
                # Generate the question for missing data
                #question_text = f"The {related_column}: {related_value}_____  information for the {main_column}: {main_value} is missing (Row ID: {row_id}). Do you want to modify the data?"
                question_text = f"The {related_column}: {related_value}_____  information for the {main_column}: {main_value} is missing. Please provide the missing answer"

            else:
                question_text = f"The {main_column}: {main_value} has {related_value} as {related_column}. Do you want to modify the data?"
                #question_text = f"The {main_column} | (Datatype: {main_column_datatype}): {main_value} has {related_value} as {related_column} | (Datatype: {related_column_datatype}). Do you want to modify the data?"
            question_count += 1
            question = {
                'type': 'confirm',
                'name': question_name,
                'message': question_text,
                'default': True,
                'data': question_data,
            }
            questions.append(question)
            

        # Add the row to processed rows to avoid duplicate questions
        processed_rows.add(index)

    print(f"Total questions generated: {question_count}")
    return questions

def generate_question_for_relationship(row, relationship, relationships_config, prediction):
    # Get main column and related column from the relationship
    main_column = relationship['main_column']
    related_column = relationship['related_column']

    # Get values for the selected columns
    main_value = row[main_column]
    related_value = row[related_column]

    # Check if both main and related values are not missing
    if not (pd.isnull(main_value) or pd.isnull(related_value)):
        # If not missing, generate the question without mentioning "missing"
        #question_text = f"The {main_column}: {main_value} has {related_value} as {related_column}. Do you want to modify the data? (Predicted: {'Yes' if prediction == 1 else 'No'})"
        question_text = f"The {main_column}: {main_value} has {related_value} as {related_column}. Do you want to modify the data?"
    else:
        # If missing, generate a question explicitly mentioning "missing"
        question_text = f"The {related_column}: {related_value}_____  information for the {main_column}: {main_value} is missing. Please provide the missing answer"

    return question_text

@app.route('/')
def index():
    # Access dynamically created model
    data = YourTable.query.all()

    # Extract relevant columns from the query result
    columns_to_include = [column.key for column in YourTable.__table__.columns]
    data_dict_list = [{col: getattr(row, col) for col in columns_to_include} for row in data]

    # Convert the list of dictionaries to a DataFrame
    dataframe = pd.DataFrame(data_dict_list)

    # Load relationships configuration from JSON file
    relationships_config = load_relationships_config()

    # Generate questions using machine learning model
    questions_ml = generate_questions_ml(dataframe, relationships_config, YourTable)

    # Ensure 'Correction_Needed' column is present before dropping it
    if 'Correction_Needed' in dataframe.columns:
        # Remove 'Correction_Needed' column from the DataFrame for the machine learning model
        dataframe_ml = dataframe.drop(columns=['Correction_Needed'])
    else:
        dataframe_ml = dataframe.copy()

    # Store the dataframe in the Flask app context for access in other routes
    app.config['DATAFRAME'] = dataframe

    # Get the first question from the list to extract the primary key (ID) for display
    first_question = questions_ml[0] if questions_ml else None
    current_row_id = first_question['data']['row_id'] if first_question else None

    # Convert the Python list to JSON using json.dumps with double quotes
    pre_rendered_questions = json.dumps([
        {
            "name": question["name"],
            "message": question["message"],
            "row_id": question['data']['row_id'],
            "mainColumn": question['data']['mainColumn'],
            "mainValue": question['data']['mainValue'],
            "relatedColumn": question['data']['relatedColumn'],
            "relatedValue": question['data']['relatedValue'],
            "mainColumnDatatype": question['data']['mainColumnDatatype'],  # Include mainColumnDatatype
            "relatedColumnDatatype": question['data']['relatedColumnDatatype'],  # Include relatedColumnDatatype
            
        } for question in questions_ml
    ], ensure_ascii=False)

    # Print the current row ID to the console
    if current_row_id is not None:
        print("")
    
    return render_template('index.html', pre_rendered_questions=pre_rendered_questions, current_row_id=current_row_id)

@app.route('/update_dialog_values', methods=['POST'])
def update_dialog_values():
    logging.debug("Received request to '/update_dialog_values'")
    data = request.get_json()
    logging.debug(f"Request data: {data}")

    try:
        row_id = data['rowId']
        main_column_name = data['mainColumn']
        related_column_name = data['relatedColumn']

        # Fetch the record to update from the database using the row ID
        record = db.session.get(YourTable, row_id)
        if record is None:
            return jsonify({"status": "error", "message": "Record not found"}), 404

        # Update the main column if a new value is provided
        if 'mainValue' in data and data['mainValue'] is not None:
            setattr(record, main_column_name, data['mainValue'])

        # Update the related column if a new value is provided
        if 'relatedValue' in data and data['relatedValue'] is not None:
            setattr(record, related_column_name, data['relatedValue'])

        db.session.commit()
        logging.info(f"Successfully updated row ID: {row_id}")
        response_data = {
            "status": "success",
            "message": "Values updated successfully",
            "updatedValues": {
                "mainValue": data.get('mainValue'),
                "relatedValue": data.get('relatedValue'),
                "rowId": row_id
            }
        }
        logging.debug(f"Response data: {response_data}")
        return jsonify(response_data), 200

    except Exception as e:
        logging.error(f"Error updating values: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Failed to update values"}), 500

def update_row_in_database(row_id, main_value, related_value):
    # Implement the logic to update the database row identified by row_id
    # with the new values for the main and related columns
    record = YourTable.query.filter_by(ID=row_id).first()
    if record:
        record.main_column = main_value  # Use the actual column name
        record.related_column = related_value  # Use the actual related column name
        db.session.commit()
    else:
        print(f"No record found with ID: {row_id}")

@app.route('/find_missing_values')
def find_missing_values():
    data = YourTable.query.all()
    dataframe = pd.DataFrame([row.__dict__ for row in data])
    relationships_config = load_relationships_config()

    # Get the current row ID from the URL parameters
    current_row_id = request.args.get('current_row_id', None)

    # Print the current row ID to the console
    if current_row_id is not None:
        print(f"From App Find Missing Function: {current_row_id}")

    missing_questions = generate_missing_data_questions(dataframe, relationships_config, current_row_id)

# Convert the Python list to JSON using json.dumps with double quotes
    pre_rendered_questions = json.dumps([
    {
        "name": question["name"],
        "message": question["message"].strip().replace('"', '\\"'),  # Escape double quotes
        "data": {
            "row_id": question.get('data', {}).get('row_id', None),  # Ensure row_id is included
            "datatype": question.get('data', {}).get('datatype', 'appropriate'),
            "mainColumn": question.get('data', {}).get('mainColumn', None),  # Add mainColumn
            "relatedColumn": question.get('data', {}).get('relatedColumn', None)  # Add relatedColumn
        }
    } for question in missing_questions
], ensure_ascii=False)

    #print(f"Pre-rendered Questions (Before Template): {pre_rendered_questions}")
    return render_template('index.html', pre_rendered_questions=pre_rendered_questions, current_row_id=current_row_id)

displayed_row_id_missing = None

def generate_missing_data_questions(dataframe, relationships_config, current_row_id):
    
    global displayed_row_id_missing
    
    missing_questions = []

    for index, row in dataframe.iterrows():
        # Check if the row ID matches the current_row_id, if provided
        if current_row_id is not None and row['ID'] != current_row_id:
            continue

        for relationship in relationships_config:
            main_column = relationship['main_column']
            related_column = relationship['related_column']

            main_value = row[main_column]
            related_value = row[related_column]
            column_datatype = determine_column_datatype(dataframe[related_column])

            if pd.isnull(related_value) or related_value == '':
                # If related value is missing, generate a question explicitly mentioning "missing"
                #question_text = f"The {related_column}: _____  information for the {main_column}: {main_value} is missing (Row ID: {row['ID']}). Please provide the missing answer."
                question_text = f"The {related_column}: _____  information for the {main_column}: {main_value} is missing. Please provide the missing answer."
                
                #question_text = f"The {related_column} information for the {main_column}: {main_value} is missing (Row ID: {row['ID']}). Please provide a {column_datatype} value."
                
                # Extract the primary key (ID) column name and value
                primary_key_column = 'ID' 
                row_id = row[primary_key_column]
                m_question_name = f'missing_{index}_{main_column}_{related_column}'
                
                question_data = {
                                    'row_id': row_id,
                                    'mainColumn': main_column,
                                    'relatedColumn': related_column,
                                    'datatype': column_datatype
                                }
                #print(f"Question Data (Before Adding to Questions): {question_data}")
                missing_questions.append({
                    'type': 'confirm',
                    'name': m_question_name,
                    'message': question_text,
                    'default': True,
                    'data': question_data
                })
                # Store the displayed row ID
                displayed_row_id_missing = row_id
    return missing_questions

def is_date(string):
    try:
        parse(string)
        return True
    except ValueError:
        return False

def determine_column_datatype(column):
    int_count = 0
    float_count = 0
    date_count = 0
    total_count = len(column)

    for value in column.dropna():  # Exclude NaN values
        if str(value).isdigit():
            int_count += 1
            continue

        try:
            float(value)
            float_count += 1
        except ValueError:
            pass

        if is_date(str(value)):
            date_count += 1

    if int_count / total_count > 0.8:
        return 'integer'
    elif float_count / total_count > 0.8:
        return 'float'
    elif date_count / total_count > 0.8:
        return 'date'
    else:
        return 'text'

def generate_flagged_value_questions(flagged_values, primary_keys, YourTable, column_datatypes):
    questions = []
    for i, flagged_value in enumerate(flagged_values):
        primary_key = primary_keys[i]
        row = YourTable.query.filter_by(ID=primary_key).first()
        if row:
            for col_name, col_value in row.__dict__.items():
                if col_value == flagged_value:
                    datatype = column_datatypes.get(col_name, 'unknown datatype')
                    rule_violation_text = (f"This value is violating the rules of the database; the expected value should be of {datatype} datatype.")
                    question_text = f"The value '{flagged_value}' in the '{col_name}' column is marked as flagged. Please provide an accurate value."
                    questions.append({
                        "row_id":primary_key,
                        "question": question_text,
                        "rule_violation": rule_violation_text,
                        "col_name": col_name,
                        "flagged_value": flagged_value,
                        "datatype": datatype
                    })
                    #print(f"Question Data (Before Adding to Questions): {questions}")
    return questions

@app.route('/show_flagged_values_questions')
def show_flagged_values_questions():
    flagged_values, primary_keys, column_datatypes = apply_rules_to_database(YourTable)
    questions = generate_flagged_value_questions(flagged_values, primary_keys, YourTable, column_datatypes)
    return jsonify({'questions': questions})

@app.route('/flagged_update_dialog_values', methods=['POST'])
def flagged_update_dialog_values():
    data = request.get_json()  # Extract data from the POST request
    print("Received data:", data)  # Debugging: Log the received data

    try:
        row_id = data.get('row_id')
        if row_id is None or row_id == '':
            print("rowId is missing or empty")
            return jsonify({"status": "error", "message": "rowId is missing or invalid"}), 400
        row_id = int(row_id)  # Now convert to int

        column_name = data.get('flaggedColumn')  # Extract the column name that needs to be updated
        new_value = data.get('flaggedValue')  # Extract the new value for the flagged column

        # Debugging: Log the extracted values
        print("Row ID:", row_id)
        print("Column Name:", column_name)
        print("New Value:", new_value)

        # Fetch the record to update from the database using the row ID
        record = db.session.query(YourTable).filter_by(ID=row_id).first()
        if record:
            # Dynamically update the specified column with the new value
            setattr(record, column_name, new_value)
            db.session.commit()  # Commit the changes to the database

            # Debugging: Log success message
            print(f"Successfully updated row {row_id}, column {column_name} with value {new_value}")
            return jsonify({"status": "success", "message": "Value updated successfully"}), 200
        else:
            # Debugging: Log record not found
            print(f"Record not found for Row ID: {row_id}")
            return jsonify({"status": "error", "message": "Record not found"}), 404

    except Exception as e:
        # Debugging: Log the error
        print(f"Error updating values: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/log_current_row_id', methods=['POST'])
def log_current_row_id():
    data = request.get_json()
    row_id = data.get('row_id', None)

    if row_id is not None:
        # Log the row ID to the server console
        print("")

    return jsonify({'success': True})

@app.route('/process_answers', methods=['POST'])
def process_answers():
    data = request.get_json()  # This will parse JSON data from the request
    print(f"Received data: {data}")

    # Validate the presence of required fields
    required_fields = ['rowId', 'mainColumn', 'relatedColumn', 'answer']
    for field in required_fields:
        if field not in data:
            print(f"{field} is missing in the request data")
            return jsonify({"status": "error", "message": f"{field} is missing"}), 400

    try:
        row_id = int(data['rowId'])  # Convert row_id to int
    except ValueError:
        print(f"Invalid row ID format: {data['rowId']}")
        return jsonify({"status": "error", "message": "Invalid row ID format"}), 400

    main_column = data['mainColumn']
    related_column = data['relatedColumn']
    user_answer = data['answer']  # User answer as a string

    # Ensure columns are valid
    valid_columns = {column.name for column in inspect(YourTable).columns}
    if main_column not in valid_columns or related_column not in valid_columns:
        print(f"Invalid column name(s): {main_column}, {related_column}")
        return jsonify({"status": "error", "message": "Invalid column name(s)"}), 400

    # Determine expected datatype for the related column
    expected_datatype = data.get('datatype', 'text')  # Default to 'text' if not specified
    user_answer = data.get('answer', None)
    print(f"Expected datatype: {expected_datatype}, Received answer: {user_answer}")


    # Convert user_answer to the expected data type
    if expected_datatype == 'integer':
        try:
            user_answer = int(user_answer)  # Convert to integer
        except ValueError:
            return jsonify({"status": "error", "message": "Invalid data type for answer. Expected integer."}), 400
    # Add additional checks and conversions for other datatypes as necessary.

    # Attempt to update the database with the converted user_answer
    try:
        return updateDatabaseWithAnswer(
            userQuery=user_answer,  # Use the converted user_answer
            YourTable=YourTable,
            rowId=row_id,  # Use the converted row_id
            relatedColumn=related_column,
            mainColumn=main_column,
            expected_datatype=expected_datatype
        )
    except Exception as e:
        print(f"Error updating the database: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500  # Internal Server Error

def is_valid_integer(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

# Function to update the database with the provided answer
def updateDatabaseWithAnswer(userQuery, YourTable, rowId, relatedColumn, mainColumn, expected_datatype):
    try:
        # Convert rowId to integer if it's not already
        rowId = int(rowId)

        # Find the primary key column variation that exists in the DataFrame
        primary_key_column = next((col for col in YourTable.__table__.columns.keys() if col.lower() == 'id'), None)
        if not primary_key_column:
            print("Primary key column not found in the table.")
            return jsonify({"status": "error", "message": "Primary key column not found"}), 404

        # Attempt to convert userQuery to the correct datatype if necessary
        if expected_datatype == 'integer':
            if not is_valid_integer(userQuery):
                print(f"Value conversion error: {userQuery} is not a valid integer")
                return jsonify({"status": "error", "message": "Value must be an integer"}), 400
            userQuery = int(userQuery)
        # Add additional checks and conversions for other datatypes (float, date, etc.) as necessary.

        # Fetch the record to update
        record = YourTable.query.filter_by(ID=rowId).first()
        if record:
            # Set the attribute value
            setattr(record, relatedColumn, userQuery)
            db.session.commit()
            print(f"Record updated: {rowId}, {relatedColumn}, {userQuery}")
            return jsonify({"status": "success"}), 200
        else:
            print(f"No record found for ID: {rowId}")
            return jsonify({"status": "error", "message": "Record not found"}), 404
    except ValueError as e:
        print(f"Value conversion error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    except Exception as e:
        print(f"Error updating the database: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/perform_database_operations')
def perform_database_operations():
    with app.app_context():
        # Use the SQLAlchemy session from the 'db' instance
        data = YourTable.query.all()
        dataframe = pd.DataFrame([row.__dict__ for row in data])
        relationships_config = load_relationships_config()
        train_classification_model(dataframe, relationships_config)

    return 'Database operations performed successfully!'

if __name__ == "__main__":
    # Apply rules to the database during application startup
    with app.app_context():
        flagged_values, primary_keys, column_datatypes = apply_rules_to_database(YourTable)

        # Store flagged values, primary keys, and column datatypes in the Flask application context
        app.config['FLAGGED_VALUES'] = flagged_values
        app.config['PRIMARY_KEYS'] = primary_keys
        app.config['COLUMN_DATATYPES'] = column_datatypes
        
        # Print the flagged values with primary keys
        print("\nShowing From App.py Flagged Values with their Primary Keys:")
        for i in range(len(flagged_values)):
            value = flagged_values[i]
            primary_key = primary_keys[i]
            print(f"ID: '{primary_key}' Value: '{value}'")

    # Run the Flask app
    app.run(debug=False, port=5001)