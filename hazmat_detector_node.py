import cv2
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from hazmat_vision.hazmat_inference import init_inference, run_frame


class HazmatCameraNode(Node):
    def __init__(self):
        super().__init__('hazmat_ros_node')

        self.declare_parameter('camera_id', 0)
        self.declare_parameter('confidence_threshold', 0.4)
        self.declare_parameter(
            'weights_path',
            '/home/student/hazmat_model/best.pt'
        )
        self.declare_parameter('device', 'cpu')

        self.camera_id = int(self.get_parameter('camera_id').value)
        self.conf_th = float(self.get_parameter('confidence_threshold').value)
        weights_path = self.get_parameter('weights_path').value
        device = self.get_parameter('device').value

        self.camera_topic = f'/cameras/raw/camera_{self.camera_id}'
        self.annotated_topic = f'/hazmat/annotated/camera_{self.camera_id}'
        self.labels_topic = f'/hazmat/labels/camera_{self.camera_id}'

        self.get_logger().info(f'Subscribing to: {self.camera_topic}')
        self.get_logger().info(f'Publishing annotated images to: {self.annotated_topic}')
        self.get_logger().info(f'Publishing labels to: {self.labels_topic}')
        self.get_logger().info(f'Loading YOLO model from: {weights_path}')

        init_inference(
            weights_path=weights_path,
            device_str=device
        )

        self.get_logger().info('YOLO inference initialized')

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10
        )

        self.annot_pub = self.create_publisher(
            Image,
            self.annotated_topic,
            10
        )

        self.labels_pub = self.create_publisher(
            String,
            self.labels_topic,
            10
        )

        self.get_logger().info('Hazmat detector node ready')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to convert ROS image to OpenCV: {e}')
            return

        try:
            annotated_frame, labels = run_frame(
                frame,
                confidence_threshold=self.conf_th
            )
        except Exception as e:
            self.get_logger().error(f'Hazmat inference failed: {e}')
            annotated_frame = frame.copy()
            labels = []

        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(
                annotated_frame,
                encoding='bgr8'
            )
            annotated_msg.header = msg.header
            self.annot_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish annotated image: {e}')

        label_text = ','.join(labels) if labels else 'none'
        self.labels_pub.publish(String(data=label_text))

        if labels:
            self.get_logger().info(f'Detected: {labels}')


def main(args=None):
    rclpy.init(args=args)
    node = HazmatCameraNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()