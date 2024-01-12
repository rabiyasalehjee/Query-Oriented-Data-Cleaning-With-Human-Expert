from sqlalchemy.orm import relationship
import json

def define_relationships(config_file_path='relationships_config.json'):
    # Load relationships configuration from JSON file
    with open(config_file_path, 'r') as file:
        relationships_config = json.load(file).get('relationships', [])

    for relationship_info in relationships_config:
        main_column = relationship_info.get("main_column")
        related_column = relationship_info.get("related_column")
        print(f"Main Column: {main_column}, Related Column: {related_column}")

        if main_column and related_column:
            main_column_name = main_column.capitalize() + 'Model'
            related_column_name = related_column.capitalize() + 'Model'
            main_model = globals().get(main_column_name)
            related_model = globals().get(related_column_name)

            if main_model and related_model:
                # Define the relationship
                relationship_name = f'{main_column.lower()}_{related_column.lower()}_relationship'
                relationship = relationship(
                    related_model,
                    primaryjoin=f'{main_model.__name__}.{main_column} == {related_model.__name__}.{related_column}',
                    backref=f'{main_model.__name__.lower()}_{related_column.lower()}_children'
                )
                setattr(main_model, relationship_name, relationship)
                print(f"Relationship defined between {main_model.__name__} and {related_model.__name__}")