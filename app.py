import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Enable wide mode
st.set_page_config(layout="wide")

# Load dataset
mpg = sns.load_dataset("mpg")

# Data preprocessing
X = mpg.drop("mpg", axis=1)
Y = mpg["mpg"]

# Handle categorical data
X['origin'] = X['origin'].astype('category')

# Data splitting
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, random_state=45)

# Set up pipelines
num_features = ['displacement', 'horsepower', 'weight', 'acceleration']
cat_features = ['origin']

# Numerical pipeline
numerical_pipeline = Pipeline(
    [("imputer", SimpleImputer()), ("std_scaler", StandardScaler())])

# Categorical pipeline with OneHotEncoder
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown='ignore'))
])

# Combining pipelines
pipeline = ColumnTransformer([
    ("numerical_pipeline", numerical_pipeline, num_features),
    ("categorical_pipeline", categorical_pipeline, cat_features)
])

# Train the model
model = LinearRegression()

# Transform the training data
X_train_tr = pipeline.fit_transform(X_train)
X_test_tr = pipeline.transform(X_test)

# Train the model
model.fit(X_train_tr, Y_train)

# Streamlit UI
st.markdown(
    "<h1 style='text-align: center; font-size: 40px;'>Car Mileage Prediction</h1>", 
    unsafe_allow_html=True
)

# Arrange inputs into two columns with larger and taller input fields
col1, col2 = st.columns(2)

with col1:
    cylinders = st.number_input(
        "Number of cylinders:", min_value=1, max_value=12, step=1,
        help="Enter the number of cylinders (e.g., 4, 6, 8)."
    )
    displacement = st.number_input(
        "Displacement (cc):", min_value=50.0, max_value=500.0, step=0.1,
        help="Enter the engine displacement in cubic centimeters."
    )
    horsepower = st.number_input(
        "Horsepower (hp):", min_value=50.0, max_value=500.0, step=0.1,
        help="Enter the horsepower of the car."
    )

with col2:
    weight = st.number_input(
        "Weight (lbs):", min_value=1000, max_value=6000, step=10,
        help="Enter the weight of the car in pounds."
    )
    acceleration = st.number_input(
        "Acceleration (0-60 mph time):", min_value=5.0, max_value=30.0, step=0.1,
        help="Enter the time taken to reach 60 mph in seconds."
    )
    model_year = st.number_input(
        "Model Year:", min_value=1900, max_value=2024, step=1,
        help="Enter the manufacturing year of the car."
    )

# Center-align the predict button
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button('Predict Mileage'):
    data = {
        'cylinders': [cylinders],
        'displacement': [displacement],
        'horsepower': [horsepower],
        'weight': [weight],
        'acceleration': [acceleration],
        'origin': ['europe'],  # Modify this for other origins like 'usa', 'japan'
        'model_year': [model_year]
    }
    df = pd.DataFrame(data)
    new_data = pipeline.transform(df)
    predicted_mileage = model.predict(new_data)
    mileage_value = predicted_mileage[0]
    mileage = round(mileage_value, 2)
    
    # Display the predicted mileage in a white rectangular box with black text
    st.markdown(
        f"""
        <div style='margin: 20px auto; padding: 20px; text-align: center; 
                    border: 2px solid #000; background-color: #fff; 
                    border-radius: 10px; width: 50%; font-size: 24px; font-weight: bold; color: #000;'>
            {mileage} Km/L
        </div>
        """, 
        unsafe_allow_html=True
    )
st.markdown("</div>", unsafe_allow_html=True)
