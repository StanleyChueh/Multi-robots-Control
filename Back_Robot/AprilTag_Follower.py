import cv2
import numpy as np
from apriltag import apriltag
import rclpy
from geometry_msgs.msg import Twist

def calculate_center(rect):
    center_x = int((rect[0][0][0] + rect[2][0][0]) / 2)
    center_y = int((rect[0][0][1] + rect[2][0][1]) / 2)
    return center_x, center_y

def estimate_distance(apparent_width, tag_size, focal_length):
    distance = (tag_size * focal_length) / apparent_width
    return distance

def label_apriltag(frame, tag_id, rect):
    # Draw bounding box
    cv2.polylines(frame, [rect], True, (0, 255, 0), 2)

    # Display AprilTag ID
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2
    text = f"ID: {tag_id}"
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    center_x, center_y = calculate_center(rect)
    text_position = (center_x - text_size[0] // 2, center_y - 10)
    cv2.putText(frame, text, text_position, font, font_scale, (0, 255, 0), font_thickness)

def april_tag_callback(frame, detector, tag_size, focal_length):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    twist = Twist()
    if len(detections) > 0:
        for detection in detections:
            rect = detection["lb-rb-rt-lt"]
            rect = np.array(rect, dtype=np.int32).reshape((-1, 1, 2))

            tag_id = detection["id"]
            center = calculate_center(rect)

            # Label AprilTag in the frame
            label_apriltag(frame, tag_id, rect)

            apparent_width = abs(rect[0][0][0] - rect[1][0][0])
            distance = estimate_distance(apparent_width, tag_size, focal_length)

            if center[0] < 80:
                twist.angular.z = 0.5  # Turn left
            elif 80 <= center[0] <= 89:
                twist.linear.x = 0.1  # Set the forward speed
            elif center[0] > 89:
                twist.angular.z = -0.5  # Turn right

    return twist

def main():
    focal_length = 125  # in pixels
    tag_size = 0.7  # size of the AprilTag in meters (adjust as needed)

    detector = apriltag("tagStandard41h12")

    rclpy.init()
    node = rclpy.create_node('apriltag_detection')
    cmd_vel_pub = node.create_publisher(Twist, '/tb3_1/cmd_vel', 10)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (160, 120))

        twist_apriltag = april_tag_callback(frame, detector, tag_size, focal_length)
        cmd_vel_pub.publish(twist_apriltag)

        cv2.imshow("AprilTag Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
