# BERD Configuration 
from pathlib import Path


MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 300
N_FEATURES = 128 # BERD features PCA dimensionality reduction
DATASET_ROW_COUNT = 50_000

PROJECT_PATH  = Path(__file__).resolve().parent
INPUT_FILE = f'{PROJECT_PATH}/raw/Train_v2.csv' 

OUTPUT_FILE = 'processed_data_128_dim.csv' 
TEXT_COLUMN = 'body' # Assuming the main text content is in a column named 'body'

# GLOBAL CONFIGURATION CONSTANTS
NUM_TOPICS = 30 # Number of topics to discover (can be tuned)
MAX_DF = 0.95 # Ignore terms that appear in more than 95% of the documents
MIN_DF = 2
