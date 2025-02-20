import datetime
import os
import json
from zoneinfo import ZoneInfo
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from my_packages.common import CodeEvaluationResult, Run

# Define project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


