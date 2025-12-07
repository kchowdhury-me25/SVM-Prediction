# ------------------------------------------------------------
# CUSTOMER CHURN PREDICTION USING SVM (PyCaret) - For PyCaret 1.x
# ------------------------------------------------------------

from pycaret.classification import *
import pandas as pd

print("Loading dataset...")

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------
data = pd.read_csv("customer_churn.csv")

print(data.head())
print(f"\n[{data.shape[0]} rows x {data.shape[1]} columns]\n")

# ------------------------------------------------------------
# 2. Clean Dataset
# ------------------------------------------------------------
if "TotalCharges" in data.columns:
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="ignore")

data = data.replace(" ", None)
data = data.replace("", None)

print("================= PYCARET MODEL TRAINING =================\n")

# ------------------------------------------------------------
# 3. Initialize PyCaret Environment (NO silent parameter)
# ------------------------------------------------------------
clf = setup(
    data=data,
    target="Churn",
    session_id=123
)

# ------------------------------------------------------------
# 4. Create SVM Model
# ------------------------------------------------------------
svm_model = create_model("svm")

# ------------------------------------------------------------
# 5. Tune SVM Model
# ------------------------------------------------------------
tuned_svm = tune_model(svm_model)

# ------------------------------------------------------------
# 6. Finalize Model
# ------------------------------------------------------------
final_svm = finalize_model(tuned_svm)

# ------------------------------------------------------------
# 7. Predict
# ------------------------------------------------------------
predictions = predict_model(final_svm)

print("\nSample Predictions:")
print(predictions.head())

# ------------------------------------------------------------
# 8. Save Model
# ------------------------------------------------------------
save_model(final_svm, "SVM_Churn_Model")

print("\nModel saved successfully.")
