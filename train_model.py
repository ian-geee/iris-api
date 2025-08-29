from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib, json, numpy as np

if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    print("X shape:", X.shape, "y shape:", y.shape)

    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", max_iter=1000, verbose=1))
    ], verbose=True)

    pipe.fit(X_train, y_train)

    clf = pipe.named_steps["clf"]
    print("classes :", clf.classes_)
    print("coefficients :\n", clf.coef_)
    print("intercept :", clf.intercept_)


    y_pred = pipe.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Proba 5 premières lignes:\n", pipe.predict_proba(X_test[:5]))

    joblib.dump(pipe, "model.joblib")
    meta = {"feature_order": feature_names, 
            "classes": load_iris().target_names.tolist(),
            "model_type": "LogReg Pipeline", 
            "model_version": "v1"
            }
    with open("model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Saved model.joblib & model_meta.json")
