import genesis as gs
import numpy as np
import os
import cv2
import json
from pathlib import Path
from scipy.spatial.transform import Rotation

# Initialize Genesis
gs.init(
    backend=gs.cuda,
    seed=None,
    precision='32',
    debug=False,
    eps=1e-12,
    logging_level=None,
    theme='dark',
    logger_verbose_time=False
)

# Create a scene
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0, 0, 1.0),
        camera_lookat=(0, 0, 0.15),
        camera_fov=30,
        max_FPS=600
    ),
    sim_options=gs.options.SimOptions(dt=0.001),
    show_viewer=False,  # Set to False for faster data capture
    show_FPS=False
)

# entities 
plane = scene.add_entity(gs.morphs.Plane())
cylinder_radius = 0.03
cylinder_pos = [0.0, 0.0, 0.05] 
cylinder = scene.add_entity(
    gs.morphs.Cylinder(
        pos=cylinder_pos,
        height=0.1,
        radius=cylinder_radius,
        collision=True,
    )
)

cam = scene.add_camera(
    pos    =(3, -1.5, 0.2),
    lookat = (0.0, 0.0, 0.09),
    res    = (1280, 960),
    fov    = 30,
    GUI    = False,
)

# Build the scene
scene.build()

# Directory where data is saved
output_dir = Path("dataset/analysis")
output_dir.mkdir(parents=True, exist_ok=True)

# Load saved data from .npy Test files
rgb = np.load(output_dir / "rgb.npy")
depth = np.load(output_dir / "depth.npy")
seg = np.load(output_dir / "segmentation_mask.npy")
normal_map = np.load(output_dir / "normal_map.npy")

#FOR TEST DATA
# Determine the cylinder's label to ocalize cylinder pixels in the segmentation mask
unique_labels = np.unique(seg)
print(f"Unique test segmentation labels: {unique_labels}")
# checking seg values or scene setup
print(scene.rigid_solver.links) # 0 is background and 1 is the cylinder
input("Enter. . .")

cylinder_label = 1  # already verified

# Create a binary mask for the cylinder
cylinder_mask = (seg == cylinder_label)
print(f"Number of cylinder pixels: {np.sum(cylinder_mask)}")

# Extract pixel coordinates of the cylinder
cylinder_pixels = np.where(cylinder_mask)
cylinder_pixel_coords = list(zip(cylinder_pixels[0], cylinder_pixels[1]))  # (y, x) pairs
print(f"First 5 cylinder pixel coordinates: {cylinder_pixel_coords[:5]}")
# Save coordinates to a .npy file (binary format, easy to load)
np.save(output_dir / "cylinder_pixel_coords.npy", cylinder_pixel_coords)
print(f"Cylinder pixel coordinates saved to: {output_dir / 'cylinder_pixel_coords.npy'}")

# Visualize the cylinder region in RGB just to confirm if we are getting the right pixel
rgb_cylinder = rgb.copy()
rgb_cylinder[~cylinder_mask] = 0  # Set non-cylinder pixels to black
cv2.imwrite(str(output_dir / "rgb_cylinder.png"), cv2.cvtColor(rgb_cylinder, cv2.COLOR_RGB2BGR))

# Save binary mask as PNG
cylinder_mask_png = (cylinder_mask * 255).astype(np.uint8)
cv2.imwrite(str(output_dir / "cylinder_mask.png"), cylinder_mask_png)

# Check if is possible to extract depth and normal values for cylinder pixels
cylinder_depth = depth[cylinder_mask]
cylinder_normals = normal_map[cylinder_mask]
print(f"Cylinder depth range: {np.min(cylinder_depth):.3f} to {np.max(cylinder_depth):.3f}")
print(f"Depth data: {cylinder_depth}")
print(f"Sample cylinder normal (first pixel): {2 * (cylinder_normals[0] - 127.5) / 255}")

print("Cylinder localization complete. Check 'rgb_cylinder.png' and 'cylinder_mask.png' for verification.")

# FOR TOP DATA
# Load saved data from .npy top files
top_rgb = np.load(output_dir / "top_rgb.npy")
top_depth = np.load(output_dir / "top_depth.npy")
top_seg = np.load(output_dir / "top_segmentation_mask.npy")
top_normal_map = np.load(output_dir / "top_normal_map.npy")
# Determine the cylinder's label to ocalize cylinder pixels in the segmentation mask
top_unique_labels = np.unique(top_seg)
print(f"Unique test segmentation labels: {top_unique_labels}")
# checking seg values or scene setup
print(scene.rigid_solver.links) # 0 is background and 1 is the cylinder
input("Enter. . .")

top_cylinder_label = 1  # already verified

# Create a binary mask for the cylinder
top_cylinder_mask = (top_seg == top_cylinder_label)
print(f"Number of cylinder pixels: {np.sum(top_cylinder_mask)}")

# Extract pixel coordinates of the cylinder
top_cylinder_pixels = np.where(top_cylinder_mask)
top_cylinder_pixel_coords = list(zip(top_cylinder_pixels[0], top_cylinder_pixels[1]))  # (y, x) pairs
print(f"First 5 cylinder pixel coordinates from Top View: {top_cylinder_pixel_coords[:5]}")
# Save coordinates to a .npy file (binary format, easy to load)
np.save(output_dir / "top_cylinder_pixel_coords.npy", top_cylinder_pixel_coords)
print(f"Cylinder pixel coordinates saved to: {output_dir / 'top_cylinder_pixel_coords.npy'}")

# Visualize the cylinder region in RGB just to confirm if we are getting the right pixel
top_rgb_cylinder = top_rgb.copy()
top_rgb_cylinder[~top_cylinder_mask] = 0  # Set non-cylinder pixels to black
cv2.imwrite(str(output_dir / "top_rgb_cylinder.png"), cv2.cvtColor(top_rgb_cylinder, cv2.COLOR_RGB2BGR))

# Save binary mask as PNG
top_cylinder_mask_png = (top_cylinder_mask * 255).astype(np.uint8)
cv2.imwrite(str(output_dir / "top_cylinder_mask.png"), top_cylinder_mask_png)

# Check if is possible to extract depth and normal values for cylinder pixels
top_cylinder_depth = top_depth[top_cylinder_mask]
top_cylinder_normals = top_normal_map[top_cylinder_mask]
print(f"Sample cylinder normal (first pixel): {2 * (top_cylinder_normals[0] - 127.5) / 255}")

print("Cylinder localization complete. Check 'top_rgb_cylinder.png' and 'top_cylinder_mask.png' for verification.")

# FOR SIDE DATA
# Load saved data from .npy side files
side_rgb = np.load(output_dir / "side_rgb.npy")
side_depth = np.load(output_dir / "side_depth.npy")
side_seg = np.load(output_dir / "side_segmentation_mask.npy")
side_normal_map = np.load(output_dir / "side_normal_map.npy")
# Determine the cylinder's label to localize cylinder pixels in the segmentation mask
side_unique_labels = np.unique(side_seg)
print(f"Unique side segmentation labels: {side_unique_labels}")
# Checking seg values or scene setup
print(scene.rigid_solver.links)  # 0 is background and 1 is the cylinder
input("Enter. . .")

side_cylinder_label = 1  # already verified

# Create a binary mask for the cylinder
side_cylinder_mask = (side_seg == side_cylinder_label)
print(f"Number of cylinder pixels: {np.sum(side_cylinder_mask)}")

# Extract pixel coordinates of the cylinder
side_cylinder_pixels = np.where(side_cylinder_mask)
side_cylinder_pixel_coords = list(zip(side_cylinder_pixels[0], side_cylinder_pixels[1]))  # (y, x) pairs
print(f"First 5 cylinder pixel coordinates from Side View: {side_cylinder_pixel_coords[:5]}")
# Save coordinates to a .npy file (binary format, easy to load)
np.save(output_dir / "side_cylinder_pixel_coords.npy", side_cylinder_pixel_coords)
print(f"Cylinder pixel coordinates saved to: {output_dir / 'side_cylinder_pixel_coords.npy'}")

# Visualize the cylinder region in RGB just to confirm if we are getting the right pixel
side_rgb_cylinder = side_rgb.copy()
side_rgb_cylinder[~side_cylinder_mask] = 0  # Set non-cylinder pixels to black
cv2.imwrite(str(output_dir / "side_rgb_cylinder.png"), cv2.cvtColor(side_rgb_cylinder, cv2.COLOR_RGB2BGR))

# Save binary mask as PNG
side_cylinder_mask_png = (side_cylinder_mask * 255).astype(np.uint8)
cv2.imwrite(str(output_dir / "side_cylinder_mask.png"), side_cylinder_mask_png)

# Check if is possible to extract depth and normal values for cylinder pixels
side_cylinder_depth = side_depth[side_cylinder_mask]
side_cylinder_normals = side_normal_map[side_cylinder_mask]
print(f"Sample cylinder normal (first pixel): {2 * (side_cylinder_normals[0] - 127.5) / 255}")

print("Cylinder localization complete. Check 'side_rgb_cylinder.png' and 'side_cylinder_mask.png' for verification.")

# FOR ISO DATA
# Load saved data from .npy isometric files
iso_rgb = np.load(output_dir / "iso_rgb.npy")
iso_depth = np.load(output_dir / "iso_depth.npy")
iso_seg = np.load(output_dir / "iso_segmentation_mask.npy")
iso_normal_map = np.load(output_dir / "iso_normal_map.npy")
# Determine the cylinder's label to localize cylinder pixels in the segmentation mask
iso_unique_labels = np.unique(iso_seg)
print(f"Unique iso segmentation labels: {iso_unique_labels}")
# Checking seg values or scene setup
print(scene.rigid_solver.links)  # 0 is background and 1 is the cylinder
input("Enter. . .")

iso_cylinder_label = 1  # already verified

# Create a binary mask for the cylinder
iso_cylinder_mask = (iso_seg == iso_cylinder_label)
print(f"Number of cylinder pixels: {np.sum(iso_cylinder_mask)}")

# Extract pixel coordinates of the cylinder
iso_cylinder_pixels = np.where(iso_cylinder_mask)
iso_cylinder_pixel_coords = list(zip(iso_cylinder_pixels[0], iso_cylinder_pixels[1]))  # (y, x) pairs
print(f"First 5 cylinder pixel coordinates from Isometric View: {iso_cylinder_pixel_coords[:5]}")
# Save coordinates to a .npy file (binary format, easy to load)
np.save(output_dir / "iso_cylinder_pixel_coords.npy", iso_cylinder_pixel_coords)
print(f"Cylinder pixel coordinates saved to: {output_dir / 'iso_cylinder_pixel_coords.npy'}")

# Visualize the cylinder region in RGB just to confirm if we are getting the right pixel
iso_rgb_cylinder = iso_rgb.copy()
iso_rgb_cylinder[~iso_cylinder_mask] = 0  # Set non-cylinder pixels to black
cv2.imwrite(str(output_dir / "iso_rgb_cylinder.png"), cv2.cvtColor(iso_rgb_cylinder, cv2.COLOR_RGB2BGR))

# Save binary mask as PNG
iso_cylinder_mask_png = (iso_cylinder_mask * 255).astype(np.uint8)
cv2.imwrite(str(output_dir / "iso_cylinder_mask.png"), iso_cylinder_mask_png)

# Check if is possible to extract depth and normal values for cylinder pixels
iso_cylinder_depth = iso_depth[iso_cylinder_mask]
iso_cylinder_normals = iso_normal_map[iso_cylinder_mask]
print(f"Sample cylinder normal (first pixel): {2 * (iso_cylinder_normals[0] - 127.5) / 255}")

print("Cylinder localization complete. Check 'iso_rgb_cylinder.png' and 'iso_cylinder_mask.png' for verification.")