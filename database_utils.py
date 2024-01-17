from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy import Table, MetaData
import pandas as pd
from flask_cors import CORS
from sqlalchemy import create_engine, MetaData


app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/query_data_cleaning'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])

migrate = Migrate(app, db)

metadata = MetaData()

class BaseModel(db.Model):
    __abstract__ = True

def create_tables():
    metadata.create_all(bind=engine)

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

def create_model_class(table_name):
    global YourTable
    with app.app_context():
        YourTable = create_model_class_internal(table_name)
    
    return YourTable

YourTable = None
create_model_class('db_messy')  # Replace 'your_table_name' with the actual table name

def load_data_from_database():
    with app.app_context():  # Enter the application context
        # Assuming YourTable is a SQLAlchemy model
        data = YourTable.query.all()

        # Extract relevant columns from the query result
        columns_to_include = [column.key for column in YourTable.__table__.columns]
        data_dict_list = [{col: getattr(row, col) for col in columns_to_include} for row in data]

        # Convert the list of dictionaries to a DataFrame
        dataframe = pd.DataFrame(data_dict_list)

    return dataframe  # Exit the application context before returning

def apply_rules_to_database(YourTable):
    with app.app_context():
        from rules import apply_rules, save_flagged_values_to_database
        dataframe = load_data_from_database()
        cleaned_dataframe, flagged_values = apply_rules(dataframe, YourTable)
        save_flagged_values_to_database(flagged_values, {}, cleaned_dataframe, YourTable)
    


