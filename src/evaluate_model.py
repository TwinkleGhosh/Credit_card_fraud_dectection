# model comparison

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def evaluate_all(models, X_test, y_test):

    names = ["Baseline", "Class Weight", "SMOTE"]

    for model, name in zip(models, names):

        print(f"\n{name} Model Results (Default Threshold = 0.5):\n")

        # default predictions
        predictions = model.predict(X_test)

        print(classification_report(y_test, predictions, zero_division=0))

        # ROC-AUC score

        probs = model.predict_proba(X_test)[:, 1]

        auc_score = roc_auc_score(y_test, probs)

        print(f"ROC-AUC Score: {auc_score:.4f}")

        # threshold tuning

        for threshold in [0.2, 0.15, 0.1]:

            print(f"\n{name} Model (Threshold = {threshold}):\n")

            tuned_predictions = (probs >= threshold).astype(int)

            print(classification_report(y_test, tuned_predictions, zero_division=0))

        print("\n" + "=" * 60)

    # feature importance

    print("\nTop Important Features (Class Weight Model):\n")

    model_weighted = models[1]  # second model

    try:

        importance = model_weighted.feature_importances_

        feature_names = X_test.columns

        feature_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importance}
        ).sort_values(by="Importance", ascending=False)

        print(feature_df.head(10))

    except:

        print("Feature importance not available for this model")

    # XGBoost feature importance

    print("\nTop Important Features (XGBoost Model):\n")

    model_xgb = models[3]  # fourth model

    try:

        importance = model_xgb.feature_importances_

        feature_names = X_test.columns

        feature_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importance}
        ).sort_values(by="Importance", ascending=False)

        print(feature_df.head(10))

    except:

        print("Feature importance not available for XGBoost")

    # confusion matrix (with threshold)


def plot_confusion_matrix(model, X_test, y_test, title, threshold=0.5):

    probs = model.predict_proba(X_test)[:, 1]

    predictions = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_test, predictions)

    plt.figure()

    sns.heatmap(cm, annot=True, fmt="d")

    plt.title(f"{title} (Threshold = {threshold})")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.show()

    # ROC-AUC score


def calculate_auc(model, X_test, y_test):

    probs = model.predict_proba(X_test)[:, 1]

    score = roc_auc_score(y_test, probs)

    print("ROC-AUC Score:", score)
