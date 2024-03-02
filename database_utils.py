from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import pandas as pd
from flask_cors import CORS
from sqlalchemy import create_engine, MetaData,Table
from sqlalchemy import inspect, Column, Float, DateTime, Integer
from datetime import datetime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
import threading

app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/query_data_cleaning'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])

migrate = Migrate(app, db)

metadata = MetaData()
model_cache = {}
lock = threading.Lock()

BaseModel = declarative_base()
BaseModel.query = db.session.query_property()

class BaseModel(db.Model):
    __abstract__ = True
    __declclassreg__ = False

def create_tables():
    metadata.create_all(bind=engine)

def create_model_class(table_name):
    with lock:
        if table_name in model_cache:
            return model_cache[table_name]

        metadata.reflect(bind=engine)
        if table_name in metadata.tables:
            table = metadata.tables[table_name]
        else:
            raise Exception(f"Table '{table_name}' does not exist.")

        class DynamicModel(BaseModel):
            __table__ = table
            __mapper_args__ = {'primary_key': [table.c.ID]}  # Adjust as needed

        model_cache[table_name] = DynamicModel
        return DynamicModel

#YourTable = None
#create_model_class('db_messy') 

def load_data_from_database(model_class):
    with app.app_context():  # Enter the application context
        # Directly use the passed model class
        data = model_class.query.all()

        # Extract relevant columns from the query result
        columns_to_include = [column.key for column in model_class.__table__.columns]
        data_dict_list = [{col: getattr(row, col) for col in columns_to_include} for row in data]

        # Convert the list of dictionaries to a DataFrame
        dataframe = pd.DataFrame(data_dict_list)

    return dataframe  # Exit the application context before returning


def create_clean_table_if_not_exists(clean_table_name='db_clean', dirty_table_name='db_messy'):
    inspector = inspect(engine)
    if clean_table_name not in inspector.get_table_names():
        # Get the dirty table's columns
        dirty_table = Table(dirty_table_name, metadata, autoload_with=engine)
        columns = [column.copy() for column in dirty_table.columns]

        # Define additional columns for the clean table
        additional_columns = [
            Column('ValidationScore', Float),
            Column('ValidatedAt', DateTime),
            Column('ValidatorId', Integer)
        ]
        columns.extend(additional_columns)

        # Create the clean table with additional metadata columns
        clean_table = Table(clean_table_name, metadata, *columns)
        clean_table.create(engine)
        print(f"Table '{clean_table_name}' created successfully.")
    else:
        print(f"Table '{clean_table_name}' already exists.")