
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(42)
num_samples = 1000
weight = np.random.normal(loc=75, scale=10, size=num_samples)
height = np.random.normal(loc=170, scale=15, size=num_samples)

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(weight, height, alpha=0.5)
plt.title('Simulation of Weight vs. Height')
plt.xlabel('Weight (kg)')
plt.ylabel('Height (cm)')
plt.grid(True)

# Add a text box with the equations
equation_text = (
    f"Weight ~ N(75, 10^2)\n"
    f"Height ~ N(170, 15^2)"
)
plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

# Save the plot
plt.savefig('simulation.png')

print("Simulation complete. Plot saved to simulation.png")
