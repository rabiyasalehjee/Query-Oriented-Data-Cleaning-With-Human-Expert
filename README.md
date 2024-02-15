# Query-Oriented Data Cleaning Project

This Flask web application is designed for cleaning and correcting data. It uses a combination of rules-based and machine-learning approaches to generate questions for users to review and correct the data.

## Features

- Dynamically creates SQLAlchemy models based on database tables.
- Checks existing data against predefined rules.
- Generates questions for users based on relationship configuration.
- Utilizes a machine learning model to predict corrections needed in the data.
- Provides a user interface for answering questions and correcting data.

## Getting Started

### Prerequisites

- Python 3.x
- Flask
- SQLAlchemy
- Flask-SQLAlchemy
- Flask-Migrate
- scikit-learn
- pandas

### Installation

1. Clone the repository or download the .zip file.
2. Run the application: python app.py
3. Open your browser and go to http://localhost:5001.

Usage
Answer the questions presented on the web interface to review and correct data.
Optionally, provide queries to correct missing data values.
Submit the form to process the answers and modify the database accordingly.

File Structure
- 'app.py': Main Flask application file containing the server and data processing logic.
- 'index.html': HTML template for the user interface.
- 'static/index_css.css': CSS styles for the user interface.
- 'relationships_config.json': JSON file containing relationships configuration.
- 'rules.py': Python script containing functions for applying rules and flagging incorrect values.
- 'database_utils.py: Flask-based utility for dynamic model creation, data loading, and rule application in a MySQL database.

### Demo
![Animated GIF](https://github.com/rabiyasalehjee/Query-Oriented-Data-Cleaning-With-Human-Expert/blob/main/Short%20Demo.gif)
