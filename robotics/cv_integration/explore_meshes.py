from stl import mesh
import numpy as np

# Path to the pipette STL file
pipette_file = "./meshes/base_link.stl"


# Load the STL file
pipette_mesh = mesh.Mesh.from_file(pipette_file)

# Extract the bounding box
min_coords = np.min(pipette_mesh.points.reshape(-1, 3), axis=0)
max_coords = np.max(pipette_mesh.points.reshape(-1, 3), axis=0)

# Calculate dimensions
dimensions = max_coords - min_coords
print(f"Pipette Dimensions (x, y, z): {dimensions}")
print(f"Bounding Box Min Coordinates: {min_coords}")
print(f"Bounding Box Max Coordinates: {max_coords}")
