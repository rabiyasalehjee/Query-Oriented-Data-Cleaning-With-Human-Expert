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
from rules import apply_rules
from flask_cors import CORS
from relationships import define_relationships
from sqlalchemy import inspect

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

        # Commit the changes to the database
        db.session.commit()
        print("Changes committed to the database")
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
    cleaned_dataframe, _ = apply_rules(dataframe, YourTable)  # Unpack the tuple correctly

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
        #print(f"Generating question for row with primary key {primary_key_column}={row_id}")

        # Generate a question for each relationship
        for relationship in relationships_config:
            main_column = relationship['main_column']
            related_column = relationship['related_column']
            question_name = f'q_{index}_ml_correct_{main_column}_{related_column}'

            # Add the row ID to the question data
            question_data = {
                'row_id': row_id,  
                'main_column': main_column,
                'related_column': related_column,
            }

            main_value = row[main_column]
            related_value = row[related_column]

            # Check if either main or related value is missing or empty
            if pd.isnull(main_value) or main_value == '' or pd.isnull(related_value) or related_value == '':
                # Generate the question for missing data
                question_text = f"The {related_column}: {related_value}_____  information for the {main_column}: {main_value} is missing (Row ID: {row_id}). Do you want to modify the data?"
            else:
                # Generate the question without mentioning "missing"
                question_text = f"The {main_column}: {main_value} has {related_value} as {related_column} (Row ID: {row_id}). Do you want to modify the data?"
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
        question_text = f"The {related_column}: {related_value}_____  information for the {main_column}: {main_value} is missing. Do you want to modify the data?"

    return question_text

@app.route('/')
def index():
    # Access your dynamically created model
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
            "message": question["message"].strip().replace('"', '\\"'),  # Escape double quotes
            "row_id": question.get('data', {}).get('row_id', None)  # Include row_id in the question
        } for question in questions_ml
    ], ensure_ascii=False)

    # Print the current row ID to the console
    if current_row_id is not None:
        print("\n")

    return render_template('index.html', pre_rendered_questions=pre_rendered_questions, current_row_id=current_row_id)


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

    # Print generated questions and their row IDs to the console
    #for question in missing_questions:
    #   print(f"Question: {question['message']}, Row ID: {question.get('data', {}).get('row_id', None)}")

    # Convert the Python list to JSON using json.dumps with double quotes
    pre_rendered_questions = json.dumps([
        {
            "name": question["name"],
            "message": question["message"].strip().replace('"', '\\"')  # Escape double quotes
        } for question in missing_questions
    ], ensure_ascii=False)

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

            if pd.isnull(related_value) or related_value == '':
                # If related value is missing, generate a question explicitly mentioning "missing"
                question_text = f"The {related_column}: _____  information for the {main_column}: {main_value} is missing (Row ID: {row['ID']}). Do you want to modify the data?"

                # Extract the primary key (ID) column name and value
                primary_key_column = 'ID' 
                row_id = row[primary_key_column]

                missing_questions.append({
                    'type': 'confirm',
                    'name': f'q_{index}_{main_column}_missing',
                    'message': question_text,
                    'default': True,
                    'data': {'row_id': row_id},  # Set the row_id in the question's data
                })
                # Store the displayed row ID
                displayed_row_id_missing = row_id
    return missing_questions

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
    try:
        data = request.get_json()  # This will parse JSON data from the request
        print(f"Received data: {data}")

        # Extract data from the JSON request, including the row ID
        row_id = int(data.get('rowId', -1))  # Replace -1 with a default value if needed
        main_column = data.get('mainColumn')
        related_column = data.get('relatedColumn')
        user_answer = data.get('answer')
        print(f"Extracted data: row_id={row_id}, main_column={main_column}, related_column={related_column}, answer={user_answer}")

        # Call the function to update the database
        updateDatabaseWithAnswer(user_answer, YourTable, row_id, related_column, main_column)

        # Return a JSON response indicating success
        return jsonify({"status": "success"})

    except Exception as e:
        print(f"Error processing JSON: {e}")
        # Return a JSON response indicating failure
        return jsonify({"status": "error", "message": str(e)})

# Function to update the database with the provided answer
def updateDatabaseWithAnswer(userQuery, YourTable, rowId, relatedColumn, mainColumn):
    try:
        # Convert rowId to integer if it's not already
        rowId = int(rowId)

        # Find the primary key column variation that exists in the DataFrame
        primary_key_column = next((col for col in YourTable.__table__.columns.keys() if col.lower() == 'id'), None)

        if primary_key_column:
            # Update the row in the database based on the found primary key column
            YourTable.query.filter(getattr(YourTable, primary_key_column) == rowId).update({relatedColumn: userQuery})

            # Commit the changes to the database
            db.session.commit()
            print(f"Changes committed to the database")
        else:
            print("Primary key column not found in the table.")

    except Exception as e:
        print(f"Error updating the database: {e}")

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
        apply_rules_to_database(YourTable)

    app.run(debug=False, port=5001)