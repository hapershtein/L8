
import numpy as np
import matplotlib.pyplot as plt
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

# Perform linear regression
regression = LinearRegression()
regression.fit(weight, height)
reg_a = regression.coef_[0][0]
reg_b = regression.intercept_[0]

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(weight, height, alpha=0.5, label='Generated Data')

# Plot the original and regression lines
plt.plot(weight, a * weight + b, color='red', linewidth=2, label='Original Line')
plt.plot(weight, regression.predict(weight), color='green', linewidth=2, label='Regression Line')

plt.title('Simulation of Height vs. Weight with Linear Regression')
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
plt.savefig('simulation2.png')

print("Simulation complete. Plot saved to simulation2.png")
