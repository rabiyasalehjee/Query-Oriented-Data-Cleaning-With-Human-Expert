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

def generate_questions(dataframe, relationships_config):
    questions = []

    # Step 1: Check Existing Data Against Rules
    existing_data_errors = check_existing_data(dataframe)
    if existing_data_errors:
        for error_message in existing_data_errors:
            question_text = f"{error_message} Do you want to correct it?"
            question = {
                'type': 'confirm',
                'name': f'q_existing_data_correct',
                'message': question_text,
                'default': True,
            }
            questions.append(question)

    # Step 2: Generate Questions for User
    for index, row in dataframe.iterrows():
        for relationship in relationships_config:
            main_column = relationship['main_column']
            related_column = relationship['related_column']

            value = row[main_column]
            related_value = row[related_column]

            if value == '':
                # Question for missing data
                question_text = f"The data in a related column is missing. It is related to '{related_column}' with value '{related_value}'. Do you want to modify the data?"
                question = {
                    'type': 'confirm',
                    'name': f'q_{index}_{main_column}_missing',
                    'message': question_text,
                    'default': True,
                }
                questions.append(question)
            else:
                # Question for existing data
                question_text = f"Is the '{main_column}' value '{value}' related to '{related_column}' with value '{related_value}' correct?"
                question = {
                    'type': 'confirm',
                    'name': f'q_{index}_{main_column}_correct',
                    'message': question_text,
                    'default': True,
                }
                questions.append(question)

    return questions

# Function to check correction needed
def check_correction_needed(row):
    """
    Check if correction is needed for a given row.
    This is a placeholder function, and you should replace it with your actual logic.
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

    # Add more checks for other columns as needed

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
    # Train the classification model
    classifier, vectorizer = train_classification_model(dataframe, relationships_config)

    questions = []

    # Generate Questions for User
    for index, row in dataframe.iterrows():
        # Selecting only the relevant text columns for feature extraction
        text_columns = [column for column in dataframe.columns if dataframe[column].dtype == 'O']
        feature_vector = vectorizer.transform([str(row[column]) for column in text_columns])

        # Predict using the trained model
        prediction = classifier.predict(feature_vector)[0]

        # Randomly generate a question based on the prediction
        question_text = random_question(row, relationships_config, prediction)
        question = {
            'type': 'confirm',
            'name': f'q_{index}_ml_correct',
            'message': question_text,
            'default': True,
        }
        questions.append(question)

    return questions


def random_question(row, relationships, prediction):
    # Randomly select a relationship
    relationship = random.choice(relationships)

    # Get main column and related column from the relationship
    main_column = relationship['main_column']
    related_column = relationship['related_column']

    # Get values for the selected columns
    main_value = row[main_column]
    related_value = row[related_column]

    # Generate a random question format
    question_formats = [
        f"The data in '{main_column}' related to '{related_column}' with value '{related_value}' is '{main_value}'. Is this correct?",
        f"Is the data '{main_value}' in '{main_column}' related to '{related_column}' with value '{related_value}' accurate?",
        f"Do you confirm that the data in '{main_column}' related to '{related_column}' with value '{related_value}' is '{main_value}'?"
    ]

    # Choose a random question format
    question_format = random.choice(question_formats)

    # Generate the final question text
    question_text = f"{question_format} (Predicted: {'Yes' if prediction == 1 else 'No'})"
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

    
    # Remove 'Correction_Needed' column from the DataFrame for the machine learning model
    dataframe_ml = dataframe.drop(columns=['Correction_Needed'])

    # Store the dataframe in the Flask app context for access in other routes
    app.config['DATAFRAME'] = dataframe

    # Generate questions based on relationships configuration
    questions_rules = generate_questions(dataframe, relationships_config)

    # Combine both sets of questions
    questions = questions_ml + questions_rules

    # Convert the Python list to JSON using json.dumps with double quotes
    pre_rendered_questions = json.dumps([
        {
            "name": question["name"],
            "message": question["message"].strip().replace('"', '\\"')  # Escape double quotes
        } for question in questions
    ], ensure_ascii=False)

    return render_template('index.html', pre_rendered_questions=pre_rendered_questions)


@app.route('/process_answers', methods=['POST'])
def process_answers():
    answers = request.form.to_dict()
    # Retrieve the dataframe from the Flask app context
    dataframe = app.config['DATAFRAME']
    # Process answers (modify the database accordingly)
    for key, answer in answers.items():
        index, column, action = key.split('_')[1:]
        index, column = int(index), str(column)
        
        if action == 'missing':
            if answer.lower() == 'no':
                # Retrieve user-provided query for missing data
                user_query = request.form.get(f'q_{index}_{column}_missing_query')
                if user_query:
                    # Append the user's query to the generated queries list
                    generatedQueries.append(user_query)
        else:
            if answer.lower() == 'no':
                print(f"Modify the value in the '{column}' column for the year {dataframe.at[index, 'Year']}.")
                # Perform the actual database modification using SQLAlchemy update statements
                # Example: WorldCup.query.filter_by(YEAR=dataframe.at[index, 'YEAR']).update({column: 'new_value'})
                # Note: You need to commit the changes to make them persistent

    return 'Answers processed successfully!'


if __name__ == "__main__":
    # Train the model during application startup
    with app.app_context():
        # Access your dynamically created model
        data = YourTable.query.all()
        dataframe = pd.DataFrame([row.__dict__ for row in data])
        relationships_config = load_relationships_config()
        train_classification_model(dataframe, relationships_config)

    app.run(debug=True)

