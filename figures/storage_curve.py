import pandas as pd
import matplotlib.pyplot as plt

# 🔹 Load Data
filename = "../scripts/catchment_565_910_volume_data.csv"  #generated while downscaling flood volume to depths. Hi-res 1 sec map is required to generate this csv
df = pd.read_csv(filename)

levee_height = 8.501230239868164  # extimated levee height for specific unit catchment

# 🔹 Create the Plot
plt.figure(figsize=(8, 6))

# Plot Inside Storage Curve
plt.plot(df['volume_inside'], df['water_surface_levels_inside'], label="Inside Storage")

# Plot Outside Storage Curve
plt.plot(df['volume_outside'], df['water_surface_levels_outside'], label="Outside Storage")

# Plot Total Storage Curve
plt.plot(df['volume_total'], df['water_surface_levels_total'], label="Total Storage", linestyle=":")

# 🔹 Add Horizontal Line for Levee Height
plt.axhline(y=levee_height, color="red", linestyle="--", linewidth=1.5, label=f"Levee Height ({levee_height:.2f} m)")

plt.xlim(left=0)  # Ensure x-axis starts at 0
plt.ylim(bottom=0)  # Ensure y-axis starts at 0


# 🔹 Labels & Legend
plt.xlabel("Storage Volume (m³)")
plt.ylabel("Water Surface Level (m)")
plt.title("Storage Curves for Inside, Outside, and Total")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("storage_curve.png", dpi=300)
