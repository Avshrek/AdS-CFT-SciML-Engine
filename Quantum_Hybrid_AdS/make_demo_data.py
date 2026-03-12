import numpy as np
import os

# Create a tiny folder for GitHub
os.makedirs("data_demo", exist_ok=True)

print("Slicing out the SOTA Golden Sample (#264)...")

# Load the massive files using memory mapping so it doesn't crash your RAM
bdy_master = np.load("data_collision_master/bdy_collision.npy", mmap_mode="r")
bulk_master = np.load("data_collision_master/bulk_collision.npy", mmap_mode="r")

# Extract ONLY sample 264 (keeping the batch dimension so shape remains identical)
bdy_tiny = bdy_master[264:265].copy()
bulk_tiny = bulk_master[264:265].copy()

# Save the tiny files
np.save("data_demo/bdy_collision.npy", bdy_tiny)
np.save("data_demo/bulk_collision.npy", bulk_tiny)

print(f"✅ Done! Demo boundary size: {os.path.getsize('data_demo/bdy_collision.npy') / 1024:.1f} KB")
print(f"✅ Done! Demo bulk size: {os.path.getsize('data_demo/bulk_collision.npy') / 1024:.1f} KB")
