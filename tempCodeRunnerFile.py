import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin

# Load dataset
mpg = sns.load_dataset("mpg")

# Data preprocessing
X = mpg.drop("mpg", axis=1)
Y = mpg["mpg"]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, random_state=45)

# Define custom transformer for company name extraction


class CompanyNameExtracter(BaseEstimator, TransformerMixin):
    def fit(self, X, Y=None):
        return self

    def transform(self, X, Y=None):
        X = X.copy()
        X.loc[:, 'name'] = X["name"].apply(self.process_string)
        return X

    @staticmethod
    def process_string(value):
        map_d = {
            'maxda': 'mazda',
            'toyouta': 'toyota',
            'vokswagen': 'vw',
            'volkswagen': 'vw'
        }
        result = value.lower().strip().split(" ")
        name = result[0]
        if name in map_d.keys():
            name = map_d[name]
        return name


# Set up pipelines
num_features = ['displacement', 'horsepower', 'weight', 'acceleration']
nominal_cat_features = ['origin']
ord_features = ['name']
pass_through_cols = ['cylinders']
drop_cols = ['model_year', 'name']

numerical_pipeline = Pipeline(
    [("imputer", SimpleImputer()), ("std_scaler", StandardScaler())])
ordinal_pipeline = Pipeline([("extract_company_name", CompanyNameExtracter()),
                             ("ordinal_encoder", OrdinalEncoder()),
                             ("std_scaling", StandardScaler())])
nominal_pipeline = Pipeline([("one_hot_encoding", OneHotEncoder())])

pipeline = ColumnTransformer([
    ("numerical_pipeline", numerical_pipeline, num_features),
    ("ordinal_pipeline", ordinal_pipeline, ord_features),
    ("nominal_pipeline", nominal_pipeline, nominal_cat_features),
    ("passing_columns", "passthrough", pass_through_cols),
    ("drop_columns", "drop", drop_cols)
])

output_cols = ['displacement', 'horsepower', 'weight',
               'acceleration', "europe", "japan", "usa", "cylinders"]

# Transform training and test data
X_train_tr = pipeline.fit_transform(X_train)
X_train_tr = pd.DataFrame(X_train_tr, columns=output_cols)
X_test_tr = pipeline.fit_transform(X_test)
X_test_tr = pd.DataFrame(X_test_tr, columns=output_cols)

# Train the model
model = LinearRegression()
model.fit(X_train_tr, Y_train)

# Define RMSE function


def rmse(y, y_hat):
    return np.sqrt(mean_squared_error(y, y_hat))


# Streamlit UI
st.title('Car Mileage Prediction')

cylinders = st.number_input(
    "Enter the number of cylinders of the car:", min_value=1, max_value=12, step=1)
displacement = st.number_input(
    "Enter the displacement of the car:", min_value=50.0, max_value=500.0, step=0.1)
horsepower = st.number_input(
    "Enter the horsepower of the car:", min_value=50.0, max_value=500.0, step=0.1)
weight = st.number_input("Enter the weight of the car:",
                         min_value=1000, max_value=6000, step=10)
acceleration = st.number_input(
    "Enter the acceleration of the car:", min_value=5.0, max_value=30.0, step=0.1)
model_year = st.number_input(
    "Enter the model year of the car:", min_value=1900, max_value=2024, step=1)
origin = st.selectbox("Enter the origin of the car:",
                      ['usa', 'japan', 'europe'])

if st.button('Predict Mileage'):
    data = {
        'cylinders': [cylinders],
        'displacement': [displacement],
        'horsepower': [horsepower],
        'weight': [weight],
        'acceleration': [acceleration],
        'model_year': [model_year],
        'origin': [origin],
        'name': ['']  # Dummy value for compatibility
    }
    df = pd.DataFrame(data)
    new_data = pipeline.transform(df)
    new_data_tr = pd.DataFrame(new_data, columns=output_cols)
    predicted_mileage = model.predict(new_data_tr)
    mileage_value = predicted_mileage[0]
    mileage = round(mileage_value, 2)
    st.write(f"Predicted Mileage: {mileage} Km/L")

# Run this script using: streamlit run app.py
