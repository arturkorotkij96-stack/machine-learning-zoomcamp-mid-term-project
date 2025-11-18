import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from model.conf import FILE_PATH, N_JOBS, Y_COL, Y_COL_RAW


def prepate_data():
    """
    Load csv file to df, fillna and split df to train/validate/test 60/20/20
    """
    df = pd.read_csv(FILE_PATH)
    df[Y_COL_RAW] = pd.to_numeric(df[Y_COL_RAW], errors="coerce")
    df = df[df[Y_COL_RAW].notna()]

    # Target variable -> comment was upvoted
    df[Y_COL] = df[Y_COL_RAW] > 0
    del df[Y_COL_RAW]
    del df["ups"]

    categorical_features = df.columns[df.dtypes == "object"].to_list()
    # We are not interested in categorical columns for our case (all features we retrieved are numerical)
    df = df.drop(columns=categorical_features)
    numerical_columns = [col for col in df.columns if col not in categorical_features and col != Y_COL]

    df[numerical_columns] = df[numerical_columns].fillna(0.0)
    
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

    return df_full_train, df_train, df_val, df_test


def split_feature_and_target(df):
    categorical_features = df.columns[df.dtypes == "object"].to_list()

    preprocessor = ColumnTransformer(
        transformers=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            remainder='passthrough', # Keep all other columns (numerical)
            verbose_feature_names_out=False # Use simple feature names
    )
    # Set the output to pandas DataFrame for ease of inspection and use
    preprocessor.set_output(transform="pandas")
    df_features = df.drop(columns=[Y_COL])
    Y = df[Y_COL]

    del df[Y_COL]
    X = preprocessor.fit_transform(df)

    return X, Y

def train_model(X_train, Y_train, X_val, Y_val, model):
    """
    This function trains model and returns auc score
    """
    model.fit(X_train, Y_train)
    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(Y_val, y_pred)

    return model, auc