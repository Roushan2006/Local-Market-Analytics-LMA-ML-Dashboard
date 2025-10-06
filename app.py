import pandas as pd
import numpy as np
import json 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from flask import Flask, render_template, request 


# Load the generated datasets (Ensure 'Collection of Data' folder exists with the CSVs)
try:
    df_businesses = pd.read_csv('Collection of Data/local_businesses.csv')
    df_users = pd.read_csv('Collection of Data/local_users.csv')
    df_interactions = pd.read_csv('Collection of Data/user_interactions.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("One or more CSV files not found. Cannot proceed with model training.")
    # Exit or handle error if data is critical
    exit() 

# --- 1. Linear Regression (df_businesses: sales_target) ---
X_business = df_businesses[['review_count', 'avg_rating', 'promo_activity']]
y_business = df_businesses['sales_target']
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_business, y_business, test_size=0.3, random_state=42
)
lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)
y_pred_lr = lr_model.predict(X_test_lr)
mse_lr = mean_squared_error(y_test_lr, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = lr_model.score(X_test_lr, y_test_lr)
lr_coeffs = {
    'Review Count': f"{lr_model.coef_[0]:.2f}",
    'Avg Rating': f"{lr_model.coef_[1]:.2f}",
    'Promo Activity': f"{lr_model.coef_[2]:.2f}"
}
lr_results = {
    'MSE': f"{mse_lr:,.2f}",
    'RMSE': f"{rmse_lr:,.2f}",
    'R2 Score': f"{r2_lr:.4f}",
    'Coefficients': lr_coeffs
}

# --- 2. Decision Tree (df_users: segment_target) ---
X_user = df_users.drop('segment_target', axis=1)
y_user = df_users['segment_target']
categorical_features_dt = ['income_bracket', 'visit_frequency', 'pref_category_1']
numerical_features_dt = ['age', 'avg_spend_3mo']
preprocessor_dt = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features_dt),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_dt)
    ], remainder='drop'
)
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_dt),
    ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42))
])
X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(
    X_user, y_user, test_size=0.3, random_state=42
)
dt_pipeline.fit(X_train_dt, y_train_dt)
y_pred_dt = dt_pipeline.predict(X_test_dt)
accuracy_dt = accuracy_score(y_test_dt, y_pred_dt)
report_dt = classification_report(y_test_dt, y_pred_dt, output_dict=True)
dt_results = {
    'Accuracy': f"{accuracy_dt:.4f}",
    'Report': report_dt
}


# --- 3. Logistic Regression (df_interactions: conversion_target) ---
X_interaction = df_interactions.drop(['conversion_target', 'interaction_id', 'user_id', 'business_id', 'timestamp'], axis=1)
y_interaction = df_interactions['conversion_target']
categorical_features_logr = ['interaction_type']
numerical_features_logr = ['distance_km', 'time_spent_sec']
preprocessor_logr = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features_logr),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_logr)
    ], remainder='drop'
)
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_logr),
    ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
])
X_train_logr, X_test_logr, y_train_logr, y_test_logr = train_test_split(
    X_interaction, y_interaction, test_size=0.3, random_state=42
)
lr_pipeline.fit(X_train_logr, y_train_logr)
y_pred_logr = lr_pipeline.predict(X_test_logr)
accuracy_logr = accuracy_score(y_test_logr, y_pred_logr)
report_logr = classification_report(y_test_logr, y_pred_logr, output_dict=True)
logr_results = {
    'Accuracy': f"{accuracy_logr:.4f}",
    'Report': report_logr
}


# --- 4. k-NN Classification (Merged Data: segment_target) ---
df_merged = df_interactions.merge(df_users, on='user_id', how='left').merge(df_businesses, on='business_id', how='left')
X_merged = df_merged.drop([
    'segment_target', 'sales_target', 'conversion_target', 
    'interaction_id', 'user_id', 'business_id', 'timestamp'
], axis=1)
y_merged = df_merged['segment_target']
numerical_features_knn = ['distance_km', 'time_spent_sec', 'age', 'avg_spend_3mo', 'price_range', 'avg_rating', 'review_count', 'promo_activity']
categorical_features_knn = ['interaction_type', 'income_bracket', 'visit_frequency', 'pref_category_1', 'category', 'city']
preprocessor_knn = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features_knn),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_knn)
    ], remainder='drop'
)
knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_knn),
    ('classifier', KNeighborsClassifier(n_neighbors=5))
])
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
    X_merged, y_merged, test_size=0.3, random_state=42
)
knn_pipeline.fit(X_train_knn, y_train_knn)
y_pred_knn = knn_pipeline.predict(X_test_knn)
accuracy_knn = accuracy_score(y_test_knn, y_pred_knn)
report_knn = classification_report(y_test_knn, y_pred_knn, output_dict=True)
knn_results = {
    'Accuracy': f"{accuracy_knn:.4f}",
    'Report': report_knn
}
# --- END of Model Training ---


# --- Flask App Setup ---

app = Flask(__name__)

@app.route('/')
def index():
    """Renders the main dashboard, passing all model results."""
    all_results = {
        'lr': lr_results,
        'dt': dt_results,
        'logr': logr_results,
        'knn': knn_results
    }
    return render_template('index.html', results=all_results)


# --- NEW PREDICTION ROUTES ---

@app.route('/predict')
def predict_form():
    """Renders the prediction input form."""
    return render_template('predict.html')


@app.route('/result', methods=['POST'])
def predict_result():
    """Processes form data and returns the prediction result."""
    
    # 1. Get data from the form
    try:
        data = {
            'review_count': float(request.form['review_count']),
            'avg_rating': float(request.form['avg_rating']),
            'promo_activity': float(request.form['promo_activity']),
            'distance_km': float(request.form['distance_km']),
            'time_spent_sec': float(request.form['time_spent_sec']),
            'interaction_type': request.form['interaction_type']
        }
    except ValueError:
        return render_template('result.html', prediction_type="Error", result="Invalid input. Please ensure all numerical fields contain numbers.")

    # Convert to DataFrame (must match the training feature names)
    new_data_lr = pd.DataFrame({
        'review_count': [data['review_count']],
        'avg_rating': [data['avg_rating']],
        'promo_activity': [data['promo_activity']]
    })
    
    new_data_logr = pd.DataFrame({
        'distance_km': [data['distance_km']],
        'time_spent_sec': [data['time_spent_sec']],
        'interaction_type': [data['interaction_type']]
    })


    # 2. Make Predictions
    
    # Linear Regression (Sales Target)
    lr_pred = lr_model.predict(new_data_lr)[0]
    
    # Logistic Regression (Conversion Target)
    # Note: We must use the full pipeline for correct preprocessing
    logr_pred_array = lr_pipeline.predict(new_data_logr)
    logr_pred = logr_pred_array[0]
    
    # Format the results
    prediction_results = {
        'lr_result': f"{lr_pred:,.2f}",
        'logr_result': "Yes (1)" if logr_pred == 1 else "No (0)",
        'input_data': data
    }

    # 3. Render results page
    return render_template('result.html', results=prediction_results)


if __name__ == '__main__':
    # Ensure you are running this with python app.py
    app.run(debug=True)