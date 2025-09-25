import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# Set parameters for the equation y = ax + b + r
a = 2.5
b = 10
num_samples = 1000

# Generate random data
np.random.seed(0)
# Weight (x) data
weight = np.random.rand(num_samples, 1) * 30 + 40  # weights between 40 and 70
# Random noise (r)
noise = np.random.randn(num_samples, 1) * 10
# Height (y) data
height = a * weight + b + noise

# Normalize the data
scaler = MinMaxScaler()
data_to_normalize = np.hstack((weight, height))
normalized_data = scaler.fit_transform(data_to_normalize)

# Perform linear regression on normalized data
regression = LinearRegression()
regression.fit(normalized_data[:, 0].reshape(-1, 1), normalized_data[:, 1])
reg_a = regression.coef_[0]
reg_b = regression.intercept_

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(normalized_data[:, 0], normalized_data[:, 1], alpha=0.5, label='Normalized Data')

# Plot the regression line
plt.plot(normalized_data[:, 0], regression.predict(normalized_data[:, 0].reshape(-1, 1)), color='green', linewidth=2, label='Regression Line')

plt.title('Normalized Simulation of Height vs. Weight with Linear Regression')
plt.xlabel('Normalized Weight')
plt.ylabel('Normalized Height')
plt.legend()
plt.grid(True)

# Add a text box with the regression equation
equation_text = f"Regression Equation: y = {reg_a:.2f}x + {reg_b:.2f}"
plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

# Save the plot
plt.savefig('simulation3.png')

print("Simulation complete. Plot saved to simulation3.png")