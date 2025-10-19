
import numpy as np
import matplotlib.pyplot as plt
import os

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

# Save the plot to a file (ensure Documentation directory exists)
output_dir = os.path.join(os.path.dirname(__file__) if '__file__' in globals() else os.getcwd(), 'Documentation')
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, 'simulation6.png')
plt.savefig(plot_path)
plt.close()

# Save results summary into a Markdown file inside Documentation
results_md_path = os.path.join(output_dir, 'Results.md')
try:
    with open(results_md_path, 'w', encoding='utf-8') as md:
        md.write('# Results: simulation6\n\n')
        md.write('## Generated plot\n\n')
        # Use relative path for image so the markdown file references the image in the same folder
        md.write('![Actual vs Predicted](simulation6.png)\n\n')

        md.write('## Inputs\n\n')
        md.write(f'- num_samples: {num_samples}\n')
        md.write(f'- num_features: {num_features}\n')
        # show first 5 true coefficients
        try:
            first_true = ', '.join(f'{float(x):.4f}' for x in true_coefficients.flatten()[:5])
        except Exception:
            first_true = 'N/A'
        md.write(f'- true_coefficients (first 5): {first_true}\n')
        md.write('- noise distribution: normal(0,1) scaled by 5\n\n')

        md.write('## Equations\n\n')
        md.write(f'- Original (excerpt): {original_eq}\n')
        md.write(f'- Regression (excerpt): {regression_eq}\n\n')

        md.write('## Estimated coefficients (first 5)\n\n')
        try:
            est_first = ', '.join(f'{float(x):.4f}' for x in estimated_coefficients.flatten()[:5])
        except Exception:
            est_first = 'N/A'
        md.write(f'- {est_first}\n\n')

        md.write('## Metrics\n\n')
        try:
            ss_res = np.sum((y - y_predicted) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot
            md.write(f'- R²: {r2:.4f}\n')
        except Exception:
            md.write('- R²: N/A\n')
    md_created = True
except Exception as e:
    md_created = False
    print(f'Warning: failed to write Results.md: {e}')

print(f"Simulation complete. Graph saved to {plot_path}")
if md_created:
    print(f"Results written to {results_md_path}")
