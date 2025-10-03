import matplotlib.pyplot as plt
import numpy as np

# Define the data
x = np.array([1, 2, 3, 4, 5, 6])  # grid numbers
y = np.array([1, 2, 4, 6, 10, 12])  # heights above main channel

# Define water surface level
water_surface_level = 8  # meters

# Color the bars
colors = ['dodgerblue' if val <= water_surface_level else 'grey' for val in y]

# Create the figure
fig, ax = plt.subplots(figsize=(7, 5))

# Plot bars
ax.bar(x, y, color=colors, edgecolor='black', width=0.9)

# Add a horizontal dashed water surface line
ax.hlines(water_surface_level, xmin=x[0]-0.5, xmax=x[-1]+0.5, colors='dodgerblue', linestyles='dashed', linewidth=2)

# Customize fonts (You can adjust font sizes here!)
ax.set_xlabel('Sorted Hi-res flddif grids', fontsize=14)
ax.set_ylabel('Height above main channel (m)', fontsize=14)
ax.set_ylim(0, 20)
ax.set_xticks(x)
ax.tick_params(axis='both', which='major', labelsize=12)  # tick label font size
ax.set_title('Flooded Area Estimation', fontsize=16)

# Optional: Remove grid
ax.grid(False)

# Layout adjustment
plt.tight_layout()

# Save the figure (optional)
plt.savefig('figure_2.png', dpi=300)


