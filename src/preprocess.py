import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. Load Dataset
# ==========================================

print("Loading dataset...")

df = pd.read_csv("data/customer_churn.csv")

print("Dataset Shape:", df.shape)


# ==========================================
# 2. Data Preprocessing
# ==========================================

print("\nPreprocessing data...")

# Convert TotalCharges from string to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove customerID because it is not useful for prediction
df = df.drop("customerID", axis=1)

# Remove rows where TotalCharges is missing
df = df.dropna(subset=["TotalCharges"])

print("Shape after preprocessing:", df.shape)


# ==========================================
# 3. Separate Features and Target
# ==========================================

X = df.drop("Churn", axis=1)

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ==========================================
# 4. Identify Columns
# ==========================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nCategorical Columns:")
print(categorical_columns)

print("\nNumerical Columns:")
print(numerical_columns)


# ==========================================
# 5. Preprocessing Pipeline
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)


# ==========================================
# 6. Create ML Model
# ==========================================

model = LogisticRegression(
    max_iter=1000
)


# ==========================================
# 7. Create Complete Pipeline
# ==========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==========================================
# 8. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ==========================================
# 9. Train Model
# ==========================================

print("\nTraining model...")

pipeline.fit(X_train, y_train)


# ==========================================
# 10. Save Trained Pipeline
# ==========================================

joblib.dump(
    pipeline,
    "models/churn_model.pkl"
)

print("\nModel saved successfully!")
print("Location: models/churn_model.pkl")


# ==========================================
# 11. Prediction
# ==========================================

y_pred = pipeline.predict(X_test)


# ==========================================
# 12. Evaluation
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred
))