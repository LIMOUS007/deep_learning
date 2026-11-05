import pandas as pd
import numpy as np
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def clean_customer_data(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Blank TotalCharges means a new customer who hasn't been billed yet, so the total is 0.
    # The imputer's median (about $1,397) would be wrong for these rows.
    missing = df["TotalCharges"].isna()
    if not (df.loc[missing, "tenure"] == 0).all():
        raise ValueError("Missing TotalCharges found for customers with nonzero tenure")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


def prepare_data(df, target_column, categorical_columns, numeric_columns, task="binary", batch_size=32, seed=42):
    if task not in ("binary", "multiclass", "regression"):
        raise ValueError(f'task must be "binary", "multiclass" or "regression", got {task!r}')
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column!r} not found")
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names found")
    y = df[target_column].to_numpy()
    target_encoder = None
    if task in ("binary", "multiclass"):
        # LabelEncoder would otherwise treat NaN as a class of its own.
        if df[target_column].isna().any():
            raise ValueError("Target contains missing values")
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
        if task == "binary" and len(target_encoder.classes_) != 2:
            raise ValueError("Binary classification requires exactly 2 classes")
        if task == "multiclass" and len(target_encoder.classes_) < 2:
            raise ValueError("Multiclass classification requires at least 2 classes")
        print("Target mapping:", dict(enumerate(target_encoder.classes_)))
    else:
        y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype="float32")
        if not np.isfinite(y).all():
            raise ValueError("Regression target contains missing or non-finite values")
    X = df.drop(columns=[target_column])
    numeric_set = set(numeric_columns)
    categorical_set = set(categorical_columns)
    feature_set = set(X.columns)
    if numeric_set & categorical_set:
        raise ValueError("A feature cannot be both numeric and categorical")
    assigned = numeric_set | categorical_set
    missing = feature_set - assigned
    invalid = assigned - feature_set
    if missing:
        raise ValueError(f"Unassigned columns: {sorted(missing)}")
    if invalid:
        raise ValueError(f"Columns not found in dataframe: {sorted(invalid)}")
    # Stratify only for classification; continuous targets can't be stratified.
    is_classification = task in ("binary", "multiclass")
    stratify = y if is_classification else None
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=stratify)
    stratify = y_temp if is_classification else None
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed, stratify=stratify)
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_columns),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_columns),
        ]
    )
    X_train = preprocessor.fit_transform(X_train).astype("float32")
    X_val = preprocessor.transform(X_val).astype("float32")
    X_test = preprocessor.transform(X_test).astype("float32")
    for name, data in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        if not np.isfinite(data).all():
            raise ValueError(f"Non-finite values found in {name}")
    joblib.dump(preprocessor, "preprocessor.pkl")
    joblib.dump(X.columns.tolist(), "input_columns.pkl")
    joblib.dump(preprocessor.get_feature_names_out(), "transformed_feature_names.pkl")
    if target_encoder is not None:
        joblib.dump(target_encoder, "target_encoder.pkl")
    print("Transformed shape:", X_train.shape)
    print("Train mean:", X_train.mean(axis=0).round(3))
    print("Train std: ", X_train.std(axis=0).round(3))
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    if task == "multiclass":
        # CrossEntropyLoss expects integer class indices with shape (N,).
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
    else:
        # BCEWithLogitsLoss and MSELoss expect float targets with shape (N, 1).
        y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
        y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    test_dataset = TabularDataset(X_test, y_test)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

    df = pd.read_csv(r"C:\coding\MACHINE\data\customer_churn.csv")
    df = df.drop(columns=['customerID'])
    print(df.shape)
    df.info()
    print(df.describe())
    print(df.dtypes)
    print(df.isna().sum())
    print(df.duplicated().sum())
    df = clean_customer_data(df)
    print(df['Churn'].value_counts())

    train_loader, val_loader, test_loader = prepare_data(df, "Churn", categorical_columns, numeric_columns)

    features, labels = next(iter(train_loader))
    print(features.shape, labels.shape)
    print(len(train_loader), len(val_loader), len(test_loader))

