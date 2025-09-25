
import numpy as np
import matplotlib.pyplot as plt

# Set parameters for the equation y = B1x + B0 + r
B1 = 2.5
B0 = 10
num_samples = 1000

# Generate random data using vector operations
np.random.seed(0)
# Weight (x) data as a column vector
weight = np.random.rand(num_samples, 1) * 30 + 40  # weights between 40 and 70

# Create the design matrix X with a column of ones for the intercept
# This is a common practice in linear algebra to represent the bias term
X = np.hstack((np.ones((num_samples, 1)), weight))

# Calculate height (y) using matrix multiplication (linear algebra)
# y = X * W + r, where W = [[B0], [B1]]
W_original = np.array([[B0], [B1]])
noise = np.random.randn(num_samples, 1) * 10
height = X @ W_original + noise

# Find the values of B0 and B1 using the normal equation
# W = (X^T * X)^-1 * X^T * y
# This is the closed-form solution for linear regression
X_transpose = X.T
W_estimated = np.linalg.inv(X_transpose @ X) @ X_transpose @ height

B0_estimated = W_estimated[0][0]
B1_estimated = W_estimated[1][0]

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(weight, height, alpha=0.5, label='Generated Data')

# Plot the regression line
plt.plot(weight, X @ W_estimated, color='green', linewidth=2, label='Regression Line')

plt.title('Simulation with Linear Algebra Regression')
plt.xlabel('Weight (kg)')
plt.ylabel('Height (cm)')
plt.legend()
plt.grid(True)

# Add a text box with the regression equation
equation_text = f"Regression Equation: y = {B1_estimated:.2f}x + {B0_estimated:.2f}"
plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

# Save the plot
plt.savefig('simulation5.png')

print("Simulation complete. Plot saved to simulation5.png")
