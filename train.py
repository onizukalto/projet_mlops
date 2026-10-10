import shutil
import sys

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

N_ESTIMATORS = 50
SEUIL_ACCURACY = 0.9          # la quality gate
TRUSTED = ["sklearn.tree._tree.Tree"]

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model", skops_trusted_types=TRUSTED)
    print(f"Accuracy : {acc:.3f}")

# Quality gate : on refuse un modèle trop faible
if acc < SEUIL_ACCURACY:
    print(f"ÉCHEC : accuracy {acc:.3f} < seuil {SEUIL_ACCURACY}")
    sys.exit(1)

# Export du modèle validé, prêt pour Docker
shutil.rmtree("model_export", ignore_errors=True)
mlflow.sklearn.save_model(model, "model_export", skops_trusted_types=TRUSTED)
print("Modèle validé et exporté dans model_export/")