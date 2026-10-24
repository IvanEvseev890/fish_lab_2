# src/config.py
import os

# Пути
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "fish_participant.csv")
SPLIT_INDICES_PATH = os.path.join(BASE_DIR, "src", "split_indices.json")

# Названия колонок
TARGET_COL = "Weight"
NUMERIC_COLS = ["Length1", "Length2", "Length3", "Height", "Width"]
CATEGORICAL_COLS = ["Species"]

# Параметры разбиения
RANDOM_STATE = 42
TEST_SIZE = 0.2