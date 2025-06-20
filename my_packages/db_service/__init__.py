import os
from pymongo import MongoClient

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
DB_NAME = os.getenv("EXPERIMENT_DB_NAME", "few_shot_experiments")
db = client[DB_NAME]

# Define project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

def get_db_connection(
        db_name: str = None,
):
    client = MongoClient(MONGO_URI)
    if db_name is None:
        DB_NAME = os.getenv("EXPERIMENT_DB_NAME")
        return client[DB_NAME]
    return client[db_name]

     