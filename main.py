from src.data_preprocessing import load_data, preprocess_data
from src.train_model import train_models
from src.evaluate_model import evaluate_all, plot_confusion_matrix, calculate_auc


def main():

    print("Loading data...")
    data = load_data("data/creditcard.csv")

    print("Preprocessing data...")
    X, y = preprocess_data(data)

    print("Training models...")
    models = train_models(X, y)

    model_baseline, model_weighted, model_smote, model_xgb, X_test, y_test = models

    print("Evaluating models...")
    evaluate_all(
        [model_baseline, model_weighted, model_smote, model_xgb], X_test, y_test
    )

    # final model

    print("\nFinal Model Evaluation (Class Weight + Threshold = 0.5):\n")

    probs = model_weighted.predict_proba(X_test)[:, 1]

    threshold = 0.5

    final_predictions = (probs >= threshold).astype(int)

    from sklearn.metrics import classification_report

    print(classification_report(y_test, final_predictions, zero_division=0))

    # final confusion matrix

    plot_confusion_matrix(model_weighted, X_test, y_test, "Final Model", threshold=0.5)

    # final ROC-AUC

    calculate_auc(model_weighted, X_test, y_test)


if __name__ == "__main__":
    main()
