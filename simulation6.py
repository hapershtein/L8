
import numpy as np
import matplotlib.pyplot as plt

# Set a seed for reproducibility
np.random.seed(42)

# --- 1. Define Simulation Parameters ---
num_samples = 1000
num_features = 50

# Generate random coefficients (b1, b2, ..., b50)
# These are the 'true' coefficients of our original equation.
true_coefficients = np.random.rand(num_features, 1) * 10

# --- 2. Generate Sample Data ---
# Generate random values for the properties (x1, x2, ..., x50)
# X is a matrix where each row is a sample and each column is a feature.
X = np.random.rand(num_samples, num_features)

# Generate a random error term 'r' for each sample
# This adds noise to our data, making it more realistic.
random_error = np.random.randn(num_samples, 1) * 5

# Calculate the output 'y' (height) using the linear equation
# y = X * true_coefficients + random_error
# This is the 'true' relationship we are trying to model.
y = X @ true_coefficients + random_error

# --- 3. Perform Linear Regression ---
# We need to add a column of ones to X for the intercept term.
# This is a standard practice in linear regression.
X_b = np.c_[np.ones((num_samples, 1)), X]

# Use the Normal Equation (via np.linalg.lstsq) to find the best coefficients
# that minimize the sum of squared errors.
# This gives us the estimated coefficients for our regression model.
estimated_coefficients, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)

# Make predictions using the estimated coefficients
y_predicted = X_b @ estimated_coefficients

# --- 4. Visualize the Results ---
plt.figure(figsize=(10, 6))

# Scatter plot of the actual data (y) vs. predicted data (y_predicted)
# If the model is good, these points should be close to a straight line.
plt.scatter(y, y_predicted, alpha=0.5, label='Actual vs. Predicted')

# Plot the ideal regression line (y=y) for reference
# This line represents a perfect prediction.
min_val = min(y.min(), y_predicted.min())
max_val = max(y.max(), y_predicted.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r-', lw=2, label='Ideal Regression Line')


# --- 5. Display Equations in a Text Box ---
# Create the text for the original and regression equations.
# We'll only show the first 3 coefficients to keep it readable.
original_eq = 'Original Eq: y = {:.2f}x1 + {:.2f}x2 + {:.2f}x3 + ... + r'.format(
    true_coefficients[0][0], true_coefficients[1][0], true_coefficients[2][0]
)
# Note: estimated_coefficients[0] is the intercept.
regression_eq = 'Regression Eq: y = {:.2f} + {:.2f}x1 + {:.2f}x2 + {:.2f}x3 + ...'.format(
    estimated_coefficients[0][0], estimated_coefficients[1][0], estimated_coefficients[2][0], estimated_coefficients[3][0]
)

# Add the text box to the plot
text_box_content = f"{original_eq}\n{regression_eq}"
plt.text(0.50, 0.15, text_box_content, transform=plt.gca().transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))


# --- 6. Finalize and Save the Plot ---
plt.title('Multiple Linear Regression Simulation')
plt.xlabel('Actual Height (y)')
plt.ylabel('Predicted Height (y_predicted)')
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig('simulation6.png')

print("Simulation complete. Graph saved to simulation6.png")
