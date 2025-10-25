from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

def evaluate_model(classifier, X_test, y_test):
    """ Evaluates the model and returns performance metrics."""
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("ROC-AUC :", roc_auc)
    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", class_report)

    return accuracy, roc_auc, conf_matrix, class_report