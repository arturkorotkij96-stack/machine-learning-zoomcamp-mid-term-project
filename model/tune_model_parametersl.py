from pathlib import Path
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text

from model.utils import prepate_data, split_feature_and_target, train_model
from model.conf import FILE_PATH, Y_COL, Y_COL_RAW

N_JOBS = 10
def grid_search(X_full_train, Y_full_train, estimator, param_grid, model_name: str = "_"):
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,  # 3-fold cross-validation
        verbose=1,
        n_jobs=N_JOBS
    )

    # --- 4. Run the Grid Search (Tuning) ---
    print(f"Starting Grid Search... {model_name}")
    grid_search.fit(X_full_train, Y_full_train)
    print(f"Grid Search complete {model_name}.")

    print(f"Best Parameters Found {model_name}: {grid_search.best_params_}")
    print(f"Best Cross-Validation AUC {model_name}: {grid_search.best_score_:.4f}")

    # --- 5. Analyze Results and Evaluate ---

    # Get the best model found by the search
    best_rf_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_auc_score = grid_search.best_score_

    return best_rf_model, best_params, best_auc_score


def fit_logistic_regression_model(X_full_train, Y_full_train):
    param_grid = {
        "C" : [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1]
    }
    estimator = LogisticRegression(solver='lbfgs', max_iter=100000)

    best_rf_model, best_params, best_auc_score = grid_search(X_full_train, Y_full_train, estimator, param_grid,  "LogisticRegression")

    return best_rf_model, best_params, best_auc_score


def fit_random_forest_model(X_full_train, Y_full_train):
    param_grid = {
        # Number of trees (Complexity/Stability)
        'n_estimators': [300, 500],
        
        # Max depth of each tree (Overfitting Control)
        'max_depth': [5, 15, 30],
        
        # Fraction of features to consider (Randomness/Variance Reduction)
        'max_features': [0.3, 0.5, 0.7, 'sqrt'], 
        
        # Minimum samples required to be at a leaf (Generalization)
        'min_samples_leaf': [3, 5, 10]
    }

    estimator = RandomForestClassifier()

    best_rf_model, best_params, best_auc_score = grid_search(X_full_train, Y_full_train, estimator, param_grid,  "RandomForestClassifier")

    return best_rf_model, best_params, best_auc_score


def fit_xgb_model(X_full_train, Y_full_train):
    param_grid = {
        # Step 1: Broad search for structure
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 5],
        
        # Step 2: Test different sampling levels
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 1.0],
        
        # Step 3: Test different learning rates (assuming you fix n_estimators high)
        'learning_rate': [0.05, 0.1],
        
        # Keep n_estimators high, or tune it first with early stopping
        'n_estimators': [300] 
    }
    # Key parameters for AUC and Imbalance:
    count_negative = (Y_full_train == 0).sum()
    count_positive = (Y_full_train == 1).sum()
    scale_pos_weight = count_negative / count_positive

    xgb_params = {
        'objective': 'binary:logistic',  # For binary classification probability output
        'eval_metric': 'auc',            # Explicitly optimize for AUC during training
        'scale_pos_weight': scale_pos_weight, # Use the calculated imbalance ratio
        # 'use_label_encoder': False,      # Suppress deprecation warning
        'n_estimators': 300,             # Number of boosting rounds (trees)
        'random_state': 42
    }
    estimator = xgb.XGBClassifier(**xgb_params)

    best_rf_model, best_params, best_auc_score = grid_search(X_full_train, Y_full_train, estimator, param_grid,  "XGBClassifier")

    return best_rf_model, best_params, best_auc_score

if __name__ == "__main__":
    df_full_train, df_train, df_val, df_test = prepate_data()
    X_full_train, Y_full_train = split_feature_and_target(df_full_train)
    X_train, Y_train = split_feature_and_target(df_train)
    X_val, Y_val = split_feature_and_target(df_val)
    X_test, Y_test = split_feature_and_target(df_test)
    # depth = 50
    scores = []
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_full_train_scaled = scaler.fit_transform(X_full_train)

    model_lg, best_params_lg, best_auc_score_lg = fit_logistic_regression_model(X_full_train_scaled, Y_full_train)
    model_lg_s, best_params_lg_s, best_auc_score_lg_s = fit_logistic_regression_model(X_full_train_scaled, Y_full_train)


    model_rf, best_params_rf, best_auc_score_rf = fit_random_forest_model(X_full_train_scaled, Y_full_train)
    model_xgb, best_params_xgb, best_auc_score_xgb = fit_xgb_model(X_full_train_scaled, Y_full_train)
 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    svm_model = SVC(
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=42
    )
    svm_model.fit(X_train_scaled, Y_train)
    y_pred = svm_model.predict_proba(X_val_scaled)[:, 1] 
    auc = roc_auc_score(Y_val, y_pred)
    print(f"svm model auc: {auc}")

    with open("model_lg.pkl", 'wb') as file:
        # 3. Save the model object to the file
        pickle.dump(model_lg, file)
    with open("model_lg_s.pkl", 'wb') as file:
        # 3. Save the model object to the file
        pickle.dump(model_lg_s, file)
    with open("model_xgb.pkl", 'wb') as file:
        # 3. Save the model object to the file
        pickle.dump(model_xgb, file)
    with open("svm_model.pkl", 'wb') as file:
        # 3. Save the model object to the file
        pickle.dump(svm_model, file)