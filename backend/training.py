import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import dump

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    recall_score,
    f1_score
)

def train_model():
    try:
        # load env file content to env vars
        load_dotenv()

        PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()

        DATASET_PATH = PROJECT_ROOT / os.getenv("DATASET_DIR") / os.getenv("DATASET_NAME")
        MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
        LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")

        TARGET_COL = os.getenv("TARGET_COL")
        TEST_SIZE  = float(os.getenv("TEST_SIZE"))
        RANDOM_STATE = int(os.getenv("RANDOM_STATE"))

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOG_PATH)
            ]    
        )

        #Load Data
        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Dataset loaded with shape: {df.shape}")

        # Seperate X and y
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        # Create a signature for each feature row to prevent duplicate leakage to test set
        row_signature = pd.util.hash_pandas_object(X, index=False)

        # Group-based split 
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        train_idx, test_idx = next(gss.split(X, y, groups=row_signature))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        logging.info(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

        # Best model params from notebook tuning
        rfc_best = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            bootstrap=True,
            ccp_alpha=0.00062,
            max_depth=3,
            max_features=0.05,
            max_samples=0.79,
            min_samples_leaf=18,
            min_samples_split=22,
            n_estimators=403
        )

        # pipeline form notebook
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", rfc_best)
            ]
        )

        pipeline.fit(X_train, y_train)
        logging.info("Model Training Completed")

        # Evaluation
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)

        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        logging.info(f"Train Accuracy: {train_acc:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}")
        logging.info(f"Test Accuracy: {test_acc:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")

        logging.info("Train Classification Report:\n"+ classification_report(y_train, y_train_pred))
        logging.info("Test Classification Report\n"+ classification_report(y_test, y_test_pred))

        # save trained model
        dump(pipeline, MODEL_PATH)
        logging.info(f"Model Saved to {MODEL_PATH}")
        logging.info("Model Training Completed.")

    except Exception as e:
        print(f"Training faliled: {e}")
        logging.exception(f"Training Script Failed: {e}")
        raise

if __name__ == "__main__":
    train_model()