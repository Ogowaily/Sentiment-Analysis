from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Train the Logistic Regression model
def train_logistic_regression(X_train_tfidf, y_train, X_test_tfidf, y_test):
    # Train the model
    model = LogisticRegression(max_iter=100)
    model.fit(X_train_tfidf, y_train)

    # Make predictions
    y_pred = model.predict(X_test_tfidf)

    # Evaluate the model
    print("🔸 Logistic Regression 🔸")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score: ", accuracy_score(y_test, y_pred))

    # Save model
    joblib.dump(model, '/content/drive/MyDrive/sentiment_project/logistic_regression_model.pkl')
