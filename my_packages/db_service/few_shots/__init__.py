import os
from db_service import client

DB_NAME = "few_shot_experiments"
db = client[DB_NAME]

# Define project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))