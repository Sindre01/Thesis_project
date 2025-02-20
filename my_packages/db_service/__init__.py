import os
from pymongo import MongoClient

# Read experiment name from environment variable or use a default
DB_NAME = os.getenv('EXPERIMENT_DB_NAME', 'default_experiment')

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
DB_NAME = os.getenv("EXPERIMENT_DB_NAME", "few_shot_experiments")
db = client[DB_NAME]

# Define project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
