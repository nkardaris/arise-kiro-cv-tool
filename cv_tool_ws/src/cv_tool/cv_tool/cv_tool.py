import argparse
import sys
from pathlib import Path
import cv2
import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Image
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge
from cv_tool_interfaces.action import Detect
from ultralytics import YOLO
from .utils import get_3d_keypoints, save_annotated_image, calc_bbox_size, TOOL_CLASS_NAMES


class CVToolActionServer(Node):
    def __init__(self, args):
        super().__init__('cv_tool_action_server')
        self.verbose = args.verbose
        self.get_logger().info("CVToolActionServer starting...")
        
        # Images & ROS2 stuff
        self.callback_group = ReentrantCallbackGroup()
        self.subscription = self.create_subscription(Image, args.rgb_topic, self.rgb_callback, 10, callback_group=self.callback_group)
        self.subscription = self.create_subscription(Image, args.depth_topic, self.depth_callback, 10, callback_group=self.callback_group)
        self.latest_rgb_frame = None
        self.latest_depth_frame = None
        self.bridge = CvBridge()
        self.rgb_counter = 0
        
        # Action server stuff
        self._action_server = ActionServer(
            self,
            Detect,
            'detect_tool',
            self.tool_detection_callback,
            callback_group=self.callback_group
        )
        
        # YOLO stuff
        model_path = f'/cv_tool_ws/src/cv_tool/cv_tool/models/{args.model}_openvino_model'
        self.get_logger().info(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path, task='detect')  # Initialize the YOLO model
        if self.verbose:
            self.output_path = Path(f'output_images_{args.model}/')
            self.output_path.mkdir(exist_ok=True)
        
        # Engineering stuff
        self.decision_buffer_size = args.buffer_size
        self.conf_thres = args.conf_thres
        self.margin_x = args.margin_x
        self.margin_y = args.margin_y
        self.allen_latest_measurment = None
        
        self.get_logger().info("CVToolActionServer ready. Waiting for goals...")



    def tool_detection_callback(self, goal_handle):
        """Executes when a new action goal is received."""
        feedback_msg = Detect.Feedback()
        result = Detect.Result()
                
        target_tool = goal_handle.request.tool_name
        self.get_logger().info(f'Received goal to detect: {target_tool}')
        if target_tool not in TOOL_CLASS_NAMES:
            self.get_logger().error(f"Requested tool '{target_tool}' is not in the known class names: {TOOL_CLASS_NAMES}")
            goal_handle.abort()
            result.success = False
            return result
        
        # Ensure we have a video feed before processing
        if self.latest_rgb_frame is None:
            self.get_logger().warn('No camera feed available.')
            goal_handle.abort()
            result.success = False
            return result
        if self.latest_depth_frame is None:
            self.get_logger().warn('No depth feed available. Proceeding with RGB only. Tool sizes may be inaccurate.')
            has_depth = False
        else:
            has_depth = True
        
        # Main processing loop
        rate = self.create_rate(15) # Process at max 15 Hz to save CPU/GPU
        frame_counter = 0
        found_counter = 0
        self.allen_latest_measurment = None

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.get_logger().info('Goal canceled by client.')
                goal_handle.canceled()
                result.success = False
                return result

            current_rgb_frame = self.latest_rgb_frame.copy()
            current_depth_frame = None
            if has_depth and self.latest_depth_frame is not None:
                current_depth_frame = self.latest_depth_frame.copy()

            # Run Inference
            class_ids, class_names, coords_list, conf_list = self.inference(current_rgb_frame)
            
            # Calculate bounding box sizes if depth data is available
            if has_depth and current_depth_frame is not None:
                bbox_sizes = []
                for i, coords in enumerate(coords_list):
                    width, height, z = calc_bbox_size(current_depth_frame, coords)
                    bbox_sizes.append((width, height, z))
                    # if self.verbose:
                    #     self.get_logger().info(f"Detection {i}: class={class_names[i]}, conf={conf_list[i]:.2f}, width={width:.2f}m, height={height:.2f}m, depth={z:.2f}m")
            else:
                bbox_sizes = [None] * len(coords_list) # Placeholder list to keep indexing consistents
            
            tool_found = False
            tool_found_and_centered = False
            target_idx = -1
            for i, name in enumerate(class_names):
                # if self.tool_in_frame(name, target_tool, current_depth_frame, coords_list[i]) and conf_list[i] > self.conf_thres:
                if self.tool_in_frame(name, target_tool, bbox_sizes[i]) and conf_list[i] > self.conf_thres:
                    found_counter += 1
                    tool_found = True
                    if self.is_tool_centered(current_rgb_frame.shape, coords_list[i], margin_x=self.margin_x, margin_y=self.margin_y) and found_counter >= self.decision_buffer_size:
                        tool_found_and_centered = True
                        target_idx = i
                        break
            
            if not tool_found:
                found_counter = 0 # reset counter if tool not found in current frame

            if tool_found_and_centered:
                self.get_logger().info(f'{target_tool} found and centered!')
                
                coords = coords_list[target_idx]
                conf = conf_list[target_idx]
                result.center, result.top_left, result.bottom_right = get_3d_keypoints(current_depth_frame, coords)
                result.success = True
                result.confidence = conf
                goal_handle.succeed()
                
                if self.verbose:
                    save_annotated_image(self.output_path, current_rgb_frame, current_depth_frame, \
                        class_names, class_ids, coords_list, conf_list, bbox_sizes, target_tool, frame_counter, \
                            extra_text='Found and Centered!')
                
                return result
            else:
                # Publish feedback if not found yet
                feedback_msg.current_status = f'Scanning frame {frame_counter} for {target_tool}...'
                goal_handle.publish_feedback(feedback_msg)
            
            if self.verbose:
                if tool_found:
                    save_annotated_image(self.output_path, current_rgb_frame, current_depth_frame, class_names, class_ids, coords_list, conf_list, bbox_sizes, target_tool, frame_counter, extra_text=f'Found but not centered #{found_counter}')
                else:
                    save_annotated_image(self.output_path, current_rgb_frame, current_depth_frame, class_names, class_ids, coords_list, conf_list, bbox_sizes, target_tool, frame_counter, extra_text='Not Found')
            
            frame_counter += 1
            
            # Sleep to maintain loop rate and yield to other threads
            rate.sleep()

    def tool_in_frame(self, detected_name, target_tool, bbox_size):
        if "allen" not in target_tool.lower():
            return target_tool.lower() in detected_name.lower()
        
        else:
            # We are looking for an allen key, but the model just says "allen" without size info.
            # We need to use depth data to verify if it's "allen_small" or "allen_large".
            if "allen" in detected_name.lower(): # We are looking at an allen key and the model also detected allen
                if bbox_size is None: # No valid depth data available
                    self.get_logger().warning("Depth data not available. Cannot verify allen key size. Relying on class name alone.")
                    return True
                
                width, height, z = bbox_size
                
                if z == 0.0 or width == 0.0 or height == 0.0 or z > 1.0 or z < 0.2:
                    self.get_logger().warning(f"Warning: Unreliable depth data for allen key (width={width:.2f}, height={height:.2f}, depth={z:.2f}). Cannot verify size")
                    if self.allen_latest_measurment is not None:
                        width, height, z = self.allen_latest_measurment
                        self.get_logger().info(f"Warning: Using latest measurement for allen key size: width={width:.2f}m, height={height:.2f}m, depth={z:.2f}m")
                    else:
                        self.get_logger().warning("Warning: No recent depth measurements available. Relying on class name alone.")
                        return True # We can't get size info --> just rely on the class name
                else: # Valid depth reading
                    self.allen_latest_measurment = (width, height, z)
                
                target_size = target_tool.lower().replace("allen", "").strip('_')
                if target_size not in ["small", "large"]: # size not specified
                    return True                    
                
                if width > 0.17 or height > 0.17:
                    detected_size = "large"
                else:
                    detected_size = "small"
                return target_size == detected_size
            
            else:
                return False
    
    
    def is_tool_centered(self, frame_shape, coords, margin_x=100, margin_y=100):
        img_h, img_w = frame_shape[:2]
        img_center_x, img_center_y = img_w / 2.0, img_h / 2.0
        
        box_center_x = (coords[0] + coords[2]) / 2.0
        box_center_y = (coords[1] + coords[3]) / 2.0
        
        condition = (abs(box_center_x - img_center_x) <= margin_x) and \
                    (abs(box_center_y - img_center_y) <= margin_y) and \
                    (coords[0] >= 10) and (coords[1] >= 10) and \
                    (coords[2] <= img_w-10) and (coords[3] <= img_h-10)
        
        return condition




    def rgb_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') # YOLO inference pipeline (self.model.predict(...) when passing a numpy array directly) expect images to be in BGR format, not RGB.
            self.latest_rgb_frame = cv_image
            self.rgb_counter += 1
        except Exception as e:
            self.get_logger().error(f'Error converting RGB image: {e}')


    def depth_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            self.latest_depth_frame = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')


    def inference(self, image):
            results = self.model.predict(
                     source=image, # camera or video path
                     device="cpu",        # explicitly use cpu
                     half=False,          # OpenVINO handles precision; keep False for CPU
                     imgsz=640,           # Lowering this (e.g., 320) will boost FPS significantly
                    #  stream=True,          # Use a generator to prevent memory bottlenecks
                    verbose=False, #self.verbose,      # Print detailed logs if verbose is set
                    agnostic_nms=True
                 )

            class_ids = []
            class_names = []
            coords_list = []
            conf_list = []
            for box in results[0].boxes:
                class_id = int(box.cls[0].item())
                class_name = results[0].names[class_id]
                coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                conf = float(box.conf[0].item()) # Confidence score
                class_ids.append(class_id)
                class_names.append(class_name)
                coords_list.append(coords)
                conf_list.append(conf)                    
            if self.verbose:    
                self.annotated_frame = results[0].plot()
            #     output_file = self.output_path.joinpath(f'annotated_{self.rgb_counter:04d}.jpg')
            #     cv2.imwrite(str(output_file), annotated_frame)
            return class_ids, class_names, coords_list, conf_list


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} must be a positive integer (>0)")
    return ivalue

def float_range_01_to_1(value):
    fvalue = float(value)
    if not (0.1 <= fvalue <= 1.0):
        raise argparse.ArgumentTypeError(f"{value} must be a float between 0.1 and 1.0")
    return fvalue

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_topic', '-rgb_topic', type=str, default='/camera/camera/color/image_raw',
                        help="RGB topic to subscribe to. Default: /camera/camera/color/image_raw")
    parser.add_argument('--depth_topic', '-depth_topic', type=str, default='/camera/camera/depth/image_rect_raw',
                    help="Depth topic to subscribe to. Default: /camera/camera/depth/image_rect_raw")
    parser.add_argument('--model', '-model', type=str, help="YOLO model", choices=['lll', '11l_int8', '11n', '11n_int8'], default='11n_int8')
    parser.add_argument('--buffer_size', '-buffer_size', type=positive_int, default=8, help='A positive integer >0 for the decision buffer size. \
                        The action is successfull only if the tool is found in buffer_size consecutive frames. Default: 8')
    parser.add_argument('--conf_thres', '-conf_thres', type=float_range_01_to_1, default=0.2, help='YOLO confidence threshold between 0.1 and 1.0. Default: 0.2')
    parser.add_argument('--margin_x', '-margin_x', type=positive_int, default=70, help='Horizontal centering tolerance in pixels (>0). Default: 70')
    parser.add_argument('--margin_y', '-margin_y', type=positive_int, default=70, help='Vertical centering tolerance in pixels (>0). Default: 70')
    parser.add_argument('--verbose', '-verbose', action='store_true', help='Enable verbosity (debug & save output images).', default=False)

    raw_args = args if args is not None else sys.argv
    app_args = remove_ros_args(args=raw_args)[1:]
    parsed_args = parser.parse_args(args=app_args)
    
    if parsed_args.verbose:
        print(f"Verbose mode enabled. Output images will be saved to: output_images_{parsed_args.model}/") 
        print("Parameters:")
        for arg, value in vars(parsed_args).items():
            print(f"  {arg}: {value}")
            
    rclpy.init(args=raw_args)
    cv_tool_action_server = CVToolActionServer(parsed_args)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(cv_tool_action_server)
    
    
    try:
        executor.spin()
        # rclpy.spin(cv_tool_action_server)
    except KeyboardInterrupt:
        pass
    finally:
        cv_tool_action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
