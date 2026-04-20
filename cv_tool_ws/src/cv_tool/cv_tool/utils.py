from math import sqrt
import cv2
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
from geometry_msgs.msg import Point

TOOL_CLASS_NAMES =  [
"allen_small",
"allen_large",
"long_nose_pliers_large",
"long_nose_pliers_small",
"wire_stripper",
"tape_measure",
"cutting_pliers_large",
"cutting_pliers_small",
"combination_wrench",
"multimeter",
"screwdriver",
"rachet"
]

# Hardcoded depth camera intrinsics from /camera/camera/depth/camera_info topic.
FX = 389.8609924316406
FY = 389.8609924316406
CX = 322.15924072265625
CY = 236.45346069335938
DEPTH_SCALE = 0.001 # Assuming 1 unit = 1 mm

def calc_bbox_size(depth_frame, coords):
    """Calculates the real-world size of a bounding box given its pixel coordinates."""
    
    x1, y1, x2, y2 = coords

    # Crop the depth image based on the bounding box coordinates
    depth_patch = depth_frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
    # Stupid OpenCV restrictions:
    # You can't run cv2.medianBlur on 16-bit uint if ksize > 5, i.e 
    # for 16-bit (CV_16U) images, the kernel size (ksize) must be 3 or 5.
    depth_patch = cv2.medianBlur(depth_patch, 5) # Apply median blur to reduce noise
    
    # Find the median value of the valid depth patch (ignoring 0 values which represent no reading)
    valid_depths = depth_patch[depth_patch > 0]
    if valid_depths.size == 0:
        return 0, 0, 0
        
    z = np.median(valid_depths) * DEPTH_SCALE
    
    # Calculate real-world width and height using pinhole camera model
    width = (x2 - x1) * z / FX
    height = (y2 - y1) * z / FY
    
    return width, height, z


def get_3d_keypoints(depth_frame, bbox_coords):
    """Given bounding box coordinates, returns the 3D coordinates of the center, top-left, and bottom-right points."""
    x1, y1, x2, y2 = bbox_coords
    
    # Crop the depth image based on the bounding box coordinates
    depth_patch = depth_frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
    
    depth_patch = cv2.medianBlur(depth_patch, 5)
    
    # Discard invalid values
    valid_depths = depth_patch[depth_patch > 0]
    if valid_depths.size == 0:
        return Point(x=0.0, y=0.0, z=0.0), Point(x=0.0, y=0.0, z=0.0), Point(x=0.0, y=0.0, z=0.0)
        
    z = np.median(valid_depths) * DEPTH_SCALE
    
    if z == 0.0:
        return Point(x=0.0, y=0.0, z=0.0), Point(x=0.0, y=0.0, z=0.0), Point(x=0.0, y=0.0, z=0.0)

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    
    center_3d = Point(x=float((center_x - CX) * z / FX), y=float((center_y - CY) * z / FY), z=float(z))
    top_left_3d = Point(x=float((x1 - CX) * z / FX), y=float((y1 - CY) * z / FY), z=float(z))
    bottom_right_3d = Point(x=float((x2 - CX) * z / FX), y=float((y2 - CY) * z / FY), z=float(z))
    
    return center_3d, top_left_3d, bottom_right_3d

def get_distance_between_pixels(depth_image, u1, v1, u2, v2):
    """
    Calculates the real-world Euclidean distance between two pixels.
    
    :param depth_image: 2D numpy array containing depth data (uint16)
    :param u1, v1: Coordinates of the first pixel
    :param u2, v2: Coordinates of the second pixel
    :return: Distance in meters, or None if a depth reading is missing.
    """
    
    # Apply median blur to the depth image to reduce noise
    blurred_depth_image = cv2.medianBlur(depth_image, 10)
        
    # Get the 3D coordinates for both pixels
    x1, y1, z1 = get_pixel_3d_coordinates(blurred_depth_image, u1, v1)
    x2, y2, z2 = get_pixel_3d_coordinates(blurred_depth_image, u2, v2)
    
    # Safety check: RealSense cameras often return 0 for invalid/unreadable pixels
    if z1 == 0.0 or z2 == 0.0:
        print("Warning: One or both pixels have no depth data (z=0).")
        return None
    
    distance = sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    return distance


def get_pixel_3d_coordinates(depth_image, u, v):
    """Deprojects a 2D pixel into 3D space."""
    
    # Ensure coordinates are integers for array indexing
    u, v = int(u), int(v)
    raw_depth = depth_image[v, u]
    z = raw_depth * DEPTH_SCALE
    
    if z == 0.0:
        return (0.0, 0.0, 0.0)
        
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    
    return (x, y, z)


def save_annotated_image(output_path, current_frame, depth_frame, class_names, class_ids, coords_list, conf_list, bbox_sizes, target_tool, counter, extra_text=None):
    annotator = Annotator(current_frame.copy(), line_width=2)
    for idx, class_name in enumerate(class_names):
        coords = coords_list[idx]
        conf = conf_list[idx]
        cls_id = class_ids[idx]
        box_color = colors(cls_id, True)
        label = f"{class_name} {conf:.2f}"
        annotator.box_label(coords, label, color=box_color)

        # Draw a small dot at the center of the bounding box.
        center_x = int((coords[0] + coords[2]) / 2)
        center_y = int((coords[1] + coords[3]) / 2)
        cv2.circle(annotator.im, (center_x, center_y), 3, box_color, -1)

        if bbox_sizes is not None:
            width, height, z = bbox_sizes[idx]
            size_label = f"{width*100:.0f}x{height*100:.0f}cm, d={z*100:.0f}cm"
            cv2.putText(annotator.im, size_label, (int(coords[0]), int(coords[1]) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    annotated_frame = annotator.result()
    if extra_text is not None:
        cv2.putText(annotated_frame, f"Goal: {target_tool}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 250, 100), 2)
        cv2.putText(annotated_frame, f"{extra_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 250, 100), 2)
    else:
        cv2.putText(annotated_frame, f"Goal: {target_tool}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 250, 100), 2)
        
    h, w = annotated_frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    cv2.drawMarker(annotated_frame, (center_x, center_y), color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    
    output_file = output_path.joinpath(f'{target_tool}_{counter:04d}.jpg')
    
    
    # Normalize depth image to 8-bit so it can be concatenated with the 8-bit BGR annotated_frame
    if depth_frame is not None:
        # Fixed scale to prevent flickering
        depth_8u = cv2.convertScaleAbs(depth_frame, alpha=255.0/2000.0)
        # Convert grayscale to 3-channel BGR so it can be concatenated with the RGB image
        depth_bgr = cv2.cvtColor(depth_8u, cv2.COLOR_GRAY2BGR)
        
        # Make sure rows (height) match
        if annotated_frame.shape[0] != depth_bgr.shape[0]:
            target_h = annotated_frame.shape[0]
            target_w = int(depth_bgr.shape[1] * (target_h / depth_bgr.shape[0]))
            depth_bgr = cv2.resize(depth_bgr, (target_w, target_h))
            
        composite_image = cv2.hconcat([annotated_frame, depth_bgr])
    else:
        composite_image = annotated_frame
    cv2.imwrite(str(output_file), composite_image)