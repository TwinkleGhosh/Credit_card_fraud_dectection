from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib


def train_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # model 1: baseline

    model_baseline = RandomForestClassifier(
        n_estimators=50, max_depth=10, n_jobs=-1, random_state=42
    )

    model_baseline.fit(X_train, y_train)

    # model 2: class weight

    model_weighted = RandomForestClassifier(
        class_weight="balanced",
        n_estimators=100,
        max_depth=10,
        n_jobs=-1,
        random_state=42,
    )

    model_weighted.fit(X_train, y_train)

    # model 3: SMOTE

    smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model_smote = RandomForestClassifier(
        n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
    )

    model_smote.fit(X_resampled, y_resampled)

    # model 4: XGBoost

    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    model_xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    model_xgb.fit(X_train, y_train)

    # save models

    joblib.dump(model_baseline, "models/model_baseline.pkl")
    joblib.dump(model_weighted, "models/model_weighted.pkl")
    joblib.dump(model_smote, "models/model_smote.pkl")
    joblib.dump(model_xgb, "models/model_xgb.pkl")

    return (model_baseline, model_weighted, model_smote, model_xgb, X_test, y_test)
