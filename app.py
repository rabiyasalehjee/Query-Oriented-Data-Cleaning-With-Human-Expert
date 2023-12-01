from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy import inspect
from sqlalchemy import Table, MetaData
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd
import pymysql
import json
import random
from rules import apply_rules

pymysql.install_as_MySQLdb()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/query_data_cleaning'
db = SQLAlchemy(app)

metadata = MetaData()

class BaseModel(db.Model):
    __abstract__ = True

YourTable = None

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

YourTable = create_model_class('db_messy')

with app.app_context():  # Create tables within the application context
    YourTable.__table__.create(bind=db.engine, checkfirst=True)

# Load relationships configuration from JSON file
def load_relationships_config():
    with open('relationships_config.json', 'r') as file:
        config = json.load(file)
    return config.get('relationships', [])

# Find related column based on relationships configuration
def find_related_column(column, relationships):
    for relationship in relationships:
        if column == relationship['main_column']:
            return relationship['related_column']
    return None

def load_data_from_database():
    # Assuming YourTable is a SQLAlchemy model
    data = YourTable.query.all()

    # Extract relevant columns from the query result
    columns_to_include = [column.key for column in YourTable.__table__.columns]
    data_dict_list = [{col: getattr(row, col) for col in columns_to_include} for row in data]

    # Convert the list of dictionaries to a DataFrame
    dataframe = pd.DataFrame(data_dict_list)

    return dataframe

# Function to apply rules to the database
def apply_rules_to_database():
    # Load data from the database
    dataframe = load_data_from_database()

    # Apply rules to clean the data
    cleaned_dataframe = apply_rules(dataframe)

    # Remove the internal SQLAlchemy column
    cleaned_dataframe = cleaned_dataframe.drop('_sa_instance_state', axis=1, errors='ignore')

    # Update the database with the cleaned data
    update_database_with_cleaned_data(cleaned_dataframe, YourTable)

    print("Data cleaning and update complete.")


# Assuming primary key variations: "ID", "id", "Id"
primary_key_variations = ["ID", "id", "Id"]

def update_database_with_cleaned_data(cleaned_dataframe, YourTable):
    for index, row in cleaned_dataframe.iterrows():
        # Find the primary key column variation that exists in the DataFrame
        primary_key_column = next((col for col in primary_key_variations if col in row.index), None)

        if primary_key_column:
            # Update the row in the database based on the found primary key column
            db.session.query(YourTable).filter(getattr(YourTable, primary_key_column) == row[primary_key_column]).update(row.to_dict())

    # Commit the changes to the database
    db.session.commit()


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
    dataframe = apply_rules(dataframe)

    # Train the classification model
    classifier, vectorizer = train_classification_model(dataframe, relationships_config)

    questions = []
    question_count = 0
    # Keep track of rows to avoid duplicate questions
    processed_rows = set()

    # Generate Questions for User
    for index, row in dataframe.iterrows():
        # Selecting all text columns for feature extraction
        text_columns = [column for column in dataframe.columns if dataframe[column].dtype == 'O']
        feature_vector = vectorizer.transform([str(row[column]) for column in text_columns])

        # Predict using the trained model
        prediction = classifier.predict(feature_vector)[0]

        if index in processed_rows:
            continue

        # Generate a question for each relationship
        for relationship in relationships_config:
            main_column = relationship['main_column']
            related_column = relationship['related_column']

            main_value = row[main_column]
            related_value = row[related_column]

            # Check if either main or related value is missing or empty
            if pd.isnull(main_value) or main_value == '' or pd.isnull(related_value) or related_value == '':
                # Generate the question for missing data
                question_text = f"The {related_column}: {related_value}_____  information for the {main_column}: {main_value} is missing. Do you want to modify the data?"
            else:
                # Generate the question without mentioning "missing"
                question_text = f"The {main_column}: {main_value} has {related_value} as {related_column}. Do you want to modify the data? (Predicted: {'Yes' if prediction == 1 else 'No'})"

            question_count += 1
            question = {
                'type': 'confirm',
                'name': f'q_{index}_ml_correct_{relationship["main_column"]}',
                'message': question_text,
                'default': True,
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
        question_text = f"The {main_column}: {main_value} has {related_value} as {related_column}. Do you want to modify the data? (Predicted: {'Yes' if prediction == 1 else 'No'})"
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

    # Convert the Python list to JSON using json.dumps with double quotes
    pre_rendered_questions = json.dumps([
        {
            "name": question["name"],
            "message": question["message"].strip().replace('"', '\\"')  # Escape double quotes
        } for question in questions_ml
    ], ensure_ascii=False)

    return render_template('index.html', pre_rendered_questions=pre_rendered_questions)

@app.route('/find_missing_values')
def find_missing_values():
    data = YourTable.query.all()
    dataframe = pd.DataFrame([row.__dict__ for row in data])
    relationships_config = load_relationships_config()
    missing_questions = generate_missing_data_questions(dataframe, relationships_config)

    # Convert the Python list to JSON using json.dumps with double quotes
    pre_rendered_questions = json.dumps([
        {
            "name": question["name"],
            "message": question["message"].strip().replace('"', '\\"')  # Escape double quotes
        } for question in missing_questions
    ], ensure_ascii=False)

    return render_template('index.html', pre_rendered_questions=pre_rendered_questions)

def generate_missing_data_questions(dataframe, relationships_config):
    missing_questions = []

    for index, row in dataframe.iterrows():
        for relationship in relationships_config:
            main_column = relationship['main_column']
            related_column = relationship['related_column']

            main_value = row[main_column]
            related_value = row[related_column]

            if pd.isnull(main_value) or main_value == '' or pd.isnull(related_value) or related_value == '':
                # Question for missing data
                question_text = f"The {related_column}: {related_value}_____ information for the {main_column}: {main_value} is missing. Do you want to modify the data?"
                missing_questions.append({
                    'type': 'confirm',
                    'name': f'q_{index}_{main_column}_missing',
                    'message': question_text,
                    'default': True,
                })

    return missing_questions

@app.route('/process_answers', methods=['POST'])
def process_answers():
    answers = request.form.to_dict()
    dataframe = app.config['DATAFRAME']

    for key, answer in answers.items():
        index, _, _ = key.split('_')[1:]
        index = int(index)

        if answer.lower() == 'no':
            print(f"Modify the data for the row at index {index}.")
            # Perform the actual data modification using SQLAlchemy update statements

    return 'Answers processed successfully!'

@app.route('/apply_rules')
def apply_rules_route():
    with app.app_context():
        apply_rules_to_database()
    return 'Rules applied to the database successfully!'

if __name__ == "__main__":
    # Apply rules to the database during application startup
    with app.app_context():
        apply_rules_to_database()

    # Train the model during application startup
    with app.app_context():
        data = YourTable.query.all()
        dataframe = pd.DataFrame([row.__dict__ for row in data])
        relationships_config = load_relationships_config()
        train_classification_model(dataframe, relationships_config)

    app.run(debug=True)