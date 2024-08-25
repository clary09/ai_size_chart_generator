# src/database.py
import sqlalchemy
import pandas as pd

class Database:
    def __init__(self, db_url):
        self.engine = sqlalchemy.create_engine(db_url)
    
    def get_all_user_data(self):
        query = "SELECT * FROM user_data"
        return pd.read_sql(query, self.engine)
