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
    up = (0, 0, 1),

)

# Build the scene
scene.build()

# Output directory
output_dir = Path("dataset/analysis")
output_dir.mkdir(parents=True, exist_ok=True)

# Number of samples
num_samples = 1  # Adjust based on dataset size needed
data_metadata = []
for i in range(num_samples):
    #Test
    # Capture RGB-D data
    render_output = cam.render(rgb=True, depth=True, segmentation=True, normal=True, colorize_seg=False)
    rgb, depth, seg, normal_map = render_output
    # Save data
    view_id = f"view_{i:02d}"
    np.save(output_dir / f"rgb.npy", rgb)
    np.save(output_dir / f"depth.npy", depth)
    np.save(output_dir / f"segmentation_mask.npy", seg)
    np.save(output_dir / f"normal_map.npy", normal_map)
    
    # Save RGB as PNG for visual verification
    cv2.imwrite(str(output_dir / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "depth.png"), cv2.cvtColor(depth, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "segmentation_mask.png"), cv2.cvtColor(seg, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "normal_map.png"), cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR))

    # Save depth as PNG (scaled to 0-255)
    depth_min, depth_max = np.min(depth), np.max(depth)
    if depth_max > depth_min:  # Avoid division by zero
        depth_scaled = (depth - depth_min) / (depth_max - depth_min) * 255
    else:
        depth_scaled = np.zeros_like(depth)
    depth_scaled = depth_scaled.astype(np.uint8)
    cv2.imwrite(str(output_dir / f"depth_{view_id}.png"), depth_scaled)
    
    # TOP VIEW
    cam.set_pose(
    pos    = (0, 0, 1),
    lookat = (0, 0, 0.09),
    up = (0, 1, 0),
    )
    
    # Capture RGB-D data
    Top_render_output = cam.render(rgb=True, depth=True, segmentation=True, normal=True, colorize_seg=False)
    Top_rgb, Top_depth, Top_seg, Top_normal_map = Top_render_output
    
    # Save data
    view_id = f"top_view_{i:02d}"
    np.save(output_dir / f"top_rgb.npy", Top_rgb)
    np.save(output_dir / f"top_depth.npy", Top_depth)
    np.save(output_dir / f"top_segmentation_mask.npy", Top_seg)
    np.save(output_dir / f"top_normal_map.npy", Top_normal_map)
    # Save RGB as PNG for visual verification
    cv2.imwrite(str(output_dir / "top_rgb.png"), cv2.cvtColor(Top_rgb, cv2.COLOR_RGB2BGR))
    # Save depth as PNG (scaled to 0-255)
    Top_depth_min, Top_depth_max = np.min(Top_depth), np.max(Top_depth)
    if Top_depth_max > Top_depth_min:  # Avoid division by zero
        Top_depth_scaled = (Top_depth - Top_depth_min) / (Top_depth_max - Top_depth_min) * 255
    else:
        Top_depth_scaled = np.zeros_like(Top_depth)
    Top_depth_scaled = Top_depth_scaled.astype(np.uint8)
    cv2.imwrite(str(output_dir / f"depth_{view_id}.png"), Top_depth_scaled)
    
    # SIDE VIEW
    cam.set_pose(
    pos    = (1, 0, 0.05),
    lookat = (0, 0, 0.09),
    up = (0, 0, 1),
    )

    # Capture RGB-D data
    side_render_output = cam.render(rgb=True, depth=True, segmentation=True, normal=True, colorize_seg=False)
    side_rgb, side_depth, side_seg, side_normal_map = side_render_output
    # Save data
    view_id = f"side_view_{i:02d}"
    np.save(output_dir / f"side_rgb.npy", side_rgb)
    np.save(output_dir / f"side_depth.npy", side_depth)
    np.save(output_dir / f"side_segmentation_mask.npy", side_seg)
    np.save(output_dir / f"side_normal_map.npy", side_normal_map)
    # Save RGB as PNG for visual verification
    cv2.imwrite(str(output_dir / "side_rgb.png"), cv2.cvtColor(side_rgb, cv2.COLOR_RGB2BGR))
    # Save depth as PNG (scaled to 0-255)
    side_depth_min, side_depth_max = np.min(side_depth), np.max(side_depth)
    if side_depth_max > side_depth_min:  # Avoid division by zero
        side_depth_scaled = (side_depth - side_depth_min) / (side_depth_max - side_depth_min) * 255
    else:
        side_depth_scaled = np.zeros_like(side_depth)
    side_depth_scaled = side_depth_scaled.astype(np.uint8)
    cv2.imwrite(str(output_dir / f"depth_{view_id}.png"), side_depth_scaled)
    
    # ISOMETRIC VIEW
    cam.set_pose(
    pos    = (1, 1, 1),
    lookat = (0, 0, 0.09),
    up = (0, 0, 1),
    )
    # Capture RGB-D data
    iso_render_output = cam.render(rgb=True, depth=True, segmentation=True, normal=True, colorize_seg=False)
    iso_rgb, iso_depth, iso_seg, iso_normal_map = iso_render_output
    # Save data
    view_id = f"iso_view_{i:02d}"
    np.save(output_dir / f"iso_rgb.npy", iso_rgb)
    np.save(output_dir / f"iso_depth.npy", iso_depth)
    np.save(output_dir / f"iso_segmentation_mask.npy", iso_seg)
    np.save(output_dir / f"iso_normal_map.npy", iso_normal_map)
    # Save RGB as PNG for visual verification
    cv2.imwrite(str(output_dir / "iso_rgb.png"), cv2.cvtColor(iso_rgb, cv2.COLOR_RGB2BGR))
    # Save depth as PNG (scaled to 0-255)
    iso_depth_min, iso_depth_max = np.min(iso_depth), np.max(iso_depth)
    if iso_depth_max > iso_depth_min:  # Avoid division by zero
        iso_depth_scaled = (iso_depth - iso_depth_min) / (iso_depth_max - iso_depth_min) * 255
    else:
        iso_depth_scaled = np.zeros_like(iso_depth)
    iso_depth_scaled = iso_depth_scaled.astype(np.uint8)
    cv2.imwrite(str(output_dir / f"depth_{view_id}.png"), iso_depth_scaled)
    
    # Store metadata
    data_metadata.append({
        "view_id": view_id,
        "camera_pos": [3, -1.5, 0.2],
        "top_camera_pos": [0, 0, 1],
        "side_camera_pos": [1, 0, 0.05],
        "iso_camera_pos": [1, 1, 1],
        "camera_lookat": [0.0, 0.0, 0.09],
        "up": [0, 0, 1],
        "top_up": [0, 1, 0],
        "fov": 30,
        "resolution": [1280, 960],
        "depth_range": [float(depth_min), float(depth_max)],
        "top_depth_range": [float(Top_depth_min), float(Top_depth_max)],
        "side_depth_range": [float(side_depth_min), float(side_depth_max)],
        "iso_depth_range": [float(iso_depth_min), float(iso_depth_max)],
    })

# Save metadata
with open(output_dir / "metadata.json", "w") as f:
    json.dump(data_metadata, f, indent=4)
print("Data capture complete. Files saved in:", output_dir)
# saved in: /home/nexus/Desktop/Genesis/examples/Exp_RigidEntity/dataset/analysis