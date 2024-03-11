#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from prius_msgs.msg import Control
from nav_msgs.msg import Odometry
import time
from collections import deque
import numpy as np

class FPSCounter:
    def __init__(self):
        self.frames = []

    def step(self):
        self.frames.append(time.monotonic())

    def get_fps(self):
        n_seconds = 5

        count = 0
        cur_time = time.monotonic()
        for f in self.frames:
            if cur_time - f < n_seconds:  # Count frames in the past n_seconds
                count += 1

        return count / n_seconds

class SolutionNode(Node):
    def __init__(self):
        super().__init__("subscriber_node")
        self.current_time = time.time()
        self.longi_integral_error = 0.0
        self.lateral_integral_error = 0.0
        self.longi_prev_error = deque([(0.0, self.current_time)] * 50)
        self.lateral_prev_error = deque([(0.0, self.current_time)] * 10)
        self.v = 0.0
        self.v_desired = 0.0
        self.heading_error = 0.0
        ### Subscriber to the image topic
        self.subscriber1 = self.create_subscription(Image,"/prius/front_camera/image_raw",self.callback,10)
        self.subscriber2 = self.create_subscription(Odometry,"/prius/odom",self.control_longitudinal,10)
        ### Publisher to the control topic
        self.publisher = self.create_publisher(Control, "/prius/control", qos_profile=10)
        self.fps_counter = FPSCounter()
        
        self.bridge = CvBridge()
        self.command = Control()
    
    def draw_fps(self, img):
        self.fps_counter.step()
        fps = self.fps_counter.get_fps()
        cv2.putText(
            img,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return img 
    
    def control_lateral(self, error:float):
        # kp = 0.015
        # ki = 0.000089
        # kd = 0.0009
        kp = 0.015
        ki = 0.000089
        kd = 0.0009
        
        de = error - self.lateral_prev_error[0][0]
        dt = self.current_time - self.lateral_prev_error[0][1]
        self.lateral_integral_error += (error * dt) / 2
        error_dot = de / dt
        u = kp * error + ki * self.lateral_integral_error + kd * error_dot
        # crosstrack_error = np.tanh(k * error)
        # self.heading_error = kd * (de / dt)
        self.command.steer = np.clip(u, -1, 1)
        # if abs(self.command.steer) < 0.1:
        #     self.command.steer = 0.0
        self.lateral_prev_error.popleft()
        self.lateral_prev_error.append((error, self.current_time))

        # self.get_logger().info(f"e: {error} tan: {np.tanh(k * error)} kd*de/dt: {kd * (de / dt)} steering: {self.command.steer}")
        self.publisher.publish(self.command)

    def control_longitudinal(self, msg:Odometry):
        # if abs(self.command.steer) > 1.5:
            # self.v_desired = min_v
        # if abs(self.heading_error) > 1.5:
            # self.v_desired = min_v
        kp = 0.2
        ki = 0.001
        kd = 0.001
        
        self.v = np.sqrt(msg.twist.twist.linear.x ** 2 + msg.twist.twist.linear.y ** 2 + msg.twist.twist.linear.z ** 2)
        # self.get_logger().info(f"v: {self.v}")

        error = self.v_desired - self.v
        de = error - self.longi_prev_error[0][0]
        dt = self.current_time - self.longi_prev_error[0][1]
        # self.get_logger().info(f"ct: {self.current_time} pe: {self.longi_prev_error[0][1]} xyt: {self.xyt[0][2]}")
        self.longi_integral_error += (error * dt) / 2
        error_dot = de / dt
        # self.get_logger().info(f"de: {de} dt: {dt} e: {error} ei: {self.longi_integral_error} ed: {error_dot}")

        u = kp * error + ki * self.longi_integral_error + kd * error_dot
        self.command.shift_gears = Control.FORWARD
        if error > 0:
            self.command.brake = 0.0
            self.command.throttle = np.clip(u, 0, 1)
            # self.get_logger().info("throttle " + str(self.command.throttle))
        else:
            self.command.brake = 1.0
            self.command.throttle = 0.0
            # self.get_logger().info("brake " + str(self.command.brake))

        self.longi_prev_error.popleft()
        self.longi_prev_error.append((error, self.current_time))
        self.publisher.publish(self.command)

    def callback(self,msg:Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv_image = self.draw_fps(cv_image)
        
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper thresholds for the red color range
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        # Combine the masks
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(cv_image, cv_image, mask=red_mask)

        # Convert the masked image to grayscale
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        rp_x = int(gray.shape[1] / 2)
        rp_y = int(gray.shape[0] * .65)
        rp_y_backup = int(gray.shape[0] * .97)
        
        left_distance = 0
        right_distance = 0
        #best 9.5 6.5 t58
        #best 9.75 t56.5
        self.v_desired = 10.5
        min_v = 7.7
        for i in range(rp_x, 0, -1):
            if gray[rp_y][i]:
                left_distance = rp_x - i
                break
        if not left_distance:
            for i in range(rp_x, 0, -1):
                if gray[rp_y_backup][i]:
                    self.get_logger().info("extereme right backup")
                    self.v_desired = min_v
                    right_distance = rp_x
                    break
            if not left_distance and not right_distance:
                self.get_logger().info("extereme left")
                self.v_desired = min_v
                left_distance = rp_x
        if not right_distance and left_distance != rp_x:
            for i in range(rp_x, gray.shape[1]):
                if gray[rp_y][i]:
                    right_distance = i - rp_x 
                    break
            if not right_distance:
                for i in range(rp_x, gray.shape[1]):
                    if gray[rp_y_backup][i]:
                        self.get_logger().info("extereme left backup")
                        self.v_desired = min_v
                        left_distance = rp_x 
                        break
                if not right_distance and left_distance != rp_x:
                    self.get_logger().info("extereme right")
                    right_distance = rp_x
                    self.v_desired = min_v
        
        steering_error = left_distance - right_distance
        # self.get_logger().info(f"ld: {left_distance} rd: {right_distance} e: {steering_error}")

        # draw steering error
        cv2.putText(
            cv_image,
            f"error: {steering_error} steering: {round(self.command.steer, 3)} v: {int(self.v)}",
            (10, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Draw reference point
        cv2.circle(cv_image, (rp_x, rp_y), 5, (255, 0, 0), -1)
        cv2.circle(cv_image, (rp_x, rp_y_backup), 5, (0, 0, 255), -1)

        # Control
        self.current_time = time.time()
        self.control_lateral(steering_error)

        #### show image
        cv2.imshow("prius_front",cv_image)
        cv2.waitKey(5)


def main():
    rclpy.init()
    node = SolutionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()