# test.py
import joblib
from sklearn.metrics import accuracy_score

def main():
    clf = joblib.load("savedmodel.pth")
    ts = joblib.load("testset.pth")
    X_test, y_test = ts['X_test'], ts['y_test']
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Loaded model test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
