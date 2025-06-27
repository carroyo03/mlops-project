import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

x = np.linspace(3,10,1000)
y = np.sin(x)

# Add noise to the data
noise = np.random.normal(0, 0.1, len(x))
y_noisy = y + noise

# Create model
model = make_pipeline(PolynomialFeatures(degree=7), LinearRegression())
model.fit(x.reshape(-1, 1), y_noisy)

# Predict the model
predictions = model.predict(x.reshape(-1, 1))

# R-squared score
r_squared = model.score(x.reshape(-1, 1), y_noisy)

# MSE
mse = np.mean((predictions - y_noisy) ** 2)

print("R-squared:", r_squared)
print("Mean Squared Error:", mse)

# Check accuracy of the model
accuracy = model.score(x.reshape(-1, 1), y)
print("Accuracy:", accuracy)