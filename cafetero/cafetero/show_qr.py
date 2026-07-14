from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from pyzbar.pyzbar import decode


class QRPosePublisher(Node):

    def __init__(self):
        super().__init__('qr_pose_publisher')

        self.bridge = CvBridge()

        self.pub = self.create_publisher(
            Float32MultiArray,
            '/qr_pose',
            10
        )

        self.create_subscription(
            Image,
            '/gripper_camera/image_raw',
            self.image_callback,
            10
        )

        self.create_subscription(
            CameraInfo,
            '/gripper_camera/camera_info',
            self.info_callback,
            10
        )

        self.camera_matrix = None
        self.dist_coeffs = None

        qr_size = 0.05
        half = qr_size / 2.0

        self.object_points = np.array([
            [-half, -half, 0],
            [ half, -half, 0],
            [ half,  half, 0],
            [-half,  half, 0]
        ], dtype=np.float32)

    def info_callback(self, msg):

        # Solo guardar una vez
        if self.camera_matrix is not None:
            return

        self.camera_matrix = np.array(
            msg.k,
            dtype=np.float32
        ).reshape(3, 3)

        self.dist_coeffs = np.array(
            msg.d,
            dtype=np.float32
        )

        self.get_logger().info(
            "CameraInfo recibido"
        )

    def image_callback(self, msg):
      
        if self.camera_matrix is None:
            print("no")
            return

        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        barcodes = decode(gray)

        poses = []

        for barcode in barcodes:

            data = barcode.data.decode('utf-8')

            pts = np.array(
                [(p.x, p.y) for p in barcode.polygon],
                dtype=np.float32
            )

            if len(pts) != 4:
                continue

            success, rvec, tvec = cv2.solvePnP(
                self.object_points,
                pts,
                self.camera_matrix,
                self.dist_coeffs
            )

            if not success:
                continue

            x, y, z = tvec.flatten()

            poses.extend([x, y, z])
            
            #DIBUJAR QR
            for i in range(4):

                p1 = tuple(pts[i].astype(int))
                p2 = tuple(pts[(i + 1) % 4].astype(int))

                cv2.line(
                    frame,
                    p1,
                    p2,
                    (0, 255, 0),
                    2
                )
                
            cv2.putText(
                frame,
                f"QR: {data}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            
            
            if data is not None:
                poses.append(100)
        
        poses_as_floats = [float(val) for val in poses]
        
        msg_out = Float32MultiArray()
        msg_out.data = poses_as_floats

        self.pub.publish(msg_out)

        cv2.imshow("QR Detection", frame)
        cv2.waitKey(1)
        
    def destroy_node(self):

        #self.cap.release()
        cv2.destroyAllWindows()

        super().destroy_node()

def main(args=None):

    rclpy.init(args=args)

    node = QRPosePublisher()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
