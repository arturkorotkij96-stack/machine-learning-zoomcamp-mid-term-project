from pathlib import Path

N_JOBS = 10

ROOT_FOLDER  = Path(__file__).resolve().parent.parent
FILE_PATH = ROOT_FOLDER / "data" / "processed" / "processed_data_128_dim.csv"
Y_COL_RAW = "score"
Y_COL = "upvoted"