
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Set parameters for the equation y = X * W + r
# W is the vector of coefficients [a, b]
a = 2.5
b = 10
W = np.array([[a], [b]])
num_samples = 1000

# Generate random data using vector operations
np.random.seed(0)
# Weight (x) data as a column vector
weight = np.random.rand(num_samples, 1) * 30 + 40  # weights between 40 and 70

# Create the design matrix X with a column of ones for the intercept
# This is a common practice in linear algebra to represent the bias term
X = np.hstack((weight, np.ones((num_samples, 1))))

# Calculate height (y) using matrix multiplication (linear algebra)
# y = X * W + r
noise = np.random.randn(num_samples, 1) * 10
height = X @ W + noise

# Perform linear regression
# The LinearRegression model solves the same equation y = X * W for W
regression = LinearRegression()
regression.fit(weight, height)
reg_a = regression.coef_[0][0]
reg_b = regression.intercept_[0]

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(weight, height, alpha=0.5, label='Generated Data')

# Plot the original and regression lines
plt.plot(weight, X @ W, color='red', linewidth=2, label='Original Line')
plt.plot(weight, regression.predict(weight), color='green', linewidth=2, label='Regression Line')

plt.title('Simulation with Vectors and Linear Algebra')
plt.xlabel('Weight (kg)')
plt.ylabel('Height (cm)')
plt.legend()
plt.grid(True)

# Add a text box with the equations
equation_text = (
    f"Original Equation: y = {a:.2f}x + {b:.2f} + r\n"
    f"Regression Equation: y = {reg_a:.2f}x + {reg_b:.2f}"
)
plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

# Save the plot
plt.savefig('simulation4.png')

print("Simulation complete. Plot saved to simulation4.png")
