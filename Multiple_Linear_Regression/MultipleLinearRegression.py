#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Step 1: Data Collection - Create a sample dataset
data = pd.DataFrame({
    'Advertising': np.random.rand(100) * 1000,  # Random advertising budget values
    'Price': np.random.rand(100) * 2000 + 500,  # Random bike price values between 500 and 2500
    'Season': np.random.choice([0, 1, 2, 3], 100),  # Random season values (0: Winter, 1: Spring, 2: Summer, 3: Fall)
    'Bikes_Sold': np.random.randint(10, 200, 100)  # Random bike sales values
})

# Display the first few rows
print(data.head())

# Step 2: Data Preprocessing
# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Check for missing values and fill them with the mean of their respective columns
print(data.isnull().sum())
data.fillna(data.mean(), inplace=True)

# Select relevant features
features = ['Advertising', 'Price', 'Season']
X = data[features]
y = data['Bikes_Sold']

# Step 3: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Generate Polynomial Features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_scaled)

# Step 5: Modeling - Implement multiple linear regression
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Step 6: Extend to polynomial regression
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Optimization - Gradient Descent with Vectorization
# This step is handled internally by scikit-learn's LinearRegression

# Step 8: Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Sales')
plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicted Sales')
plt.title('Actual vs Predicted Bike Sales')
plt.xlabel('Instance')
plt.ylabel('Number of Bikes Sold')
plt.legend()
plt.show()

# Step 9: Deployment
# Save the model and the scaler
joblib.dump(model, 'bike_sales_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(poly, 'poly_features.pkl')

# To make future predictions, you would load the model and scaler like this:
# model = joblib.load('bike_sales_model.pkl')
# scaler = joblib.load('scaler.pkl')
# poly = joblib.load('poly_features.pkl')

# Example of making a prediction with new data
new_data = np.array([[150, 2000, 1]])  # New data for 'Advertising', 'Price', 'Season'
new_data_scaled = scaler.transform(new_data)
new_data_poly = poly.transform(new_data_scaled)
predicted_sales = model.predict(new_data_poly)
print("Predicted number of bikes sold:", predicted_sales)


# In[ ]:




