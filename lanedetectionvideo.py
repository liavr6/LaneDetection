import cv2
import numpy as np
from skimage.measure import LineModelND, ransac
import csv
import os

# Parameters for smoothing and frame skipping
history_length = 5
frame_skip = 2  # Process every 2nd frame
left_line_history = []
right_line_history = []
save_visualizations = True  # Save processed frames for analysis
output_folder = "output_frames"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize trapezoid and mouse functionality
trapezoid_points = [(0, 0), (0, 0), (0, 0), (0, 0)]
dragging = -1
original_points = []

def initialize_trapezoid_points(width, height):
    global trapezoid_points, original_points
    roi_y_top = int(height * 0.6)
    roi_y_bottom = height
    left_top_x = int(width * 0.4)
    right_top_x = int(width * 0.6)
    trapezoid_points = [(0, roi_y_bottom), (width, roi_y_bottom), (right_top_x, roi_y_top), (left_top_x, roi_y_top)]
    original_points = trapezoid_points.copy()

def mouse_callback(event, x, y, flags, param):
    global dragging, trapezoid_points
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, point in enumerate(trapezoid_points):
            if abs(x - point[0]) < 10 and abs(y - point[1]) < 10:
                dragging = i
                break
    elif event == cv2.EVENT_MOUSEMOVE and dragging != -1:
        trapezoid_points[dragging] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = -1

# Functions for clamping and extending within the trapezoid
def clamp_within_trapezoid(x, y, trapezoid):
    left_x = trapezoid[3][0] + (trapezoid[0][0] - trapezoid[3][0]) * (y - trapezoid[3][1]) / (trapezoid[0][1] - trapezoid[3][1])
    right_x = trapezoid[2][0] + (trapezoid[1][0] - trapezoid[2][0]) * (y - trapezoid[2][1]) / (trapezoid[1][1] - trapezoid[2][1])
    return max(int(left_x), min(int(right_x), x))

def extend_line_within_trapezoid(line_x, line_y, y1, y2, trapezoid):
    if line_x is None or line_y is None:
        return None, None
    slope = (line_y[1] - line_y[0]) / (line_x[1] - line_x[0]) if (line_x[1] - line_x[0]) != 0 else 0
    intercept = line_y[0] - slope * line_x[0]
    new_x1 = int((y1 - intercept) / slope) if slope != 0 else line_x[0]
    new_x2 = int((y2 - intercept) / slope) if slope != 0 else line_x[1]
    new_x1 = clamp_within_trapezoid(new_x1, y1, trapezoid)
    new_x2 = clamp_within_trapezoid(new_x2, y2, trapezoid)
    return [new_x1, new_x2], [y1, y2]

def calculate_lane_center_and_steering(frame, smoothed_left_x, smoothed_right_x):
    height, width = frame.shape[:2]
    
    if smoothed_left_x is None or smoothed_right_x is None:
        return None, None

    left_x_bottom = smoothed_left_x[1]
    right_x_bottom = smoothed_right_x[1]
    lane_center = int((left_x_bottom + right_x_bottom) / 2)

    car_position = width // 2
    offset = car_position - lane_center
    steering_angle = offset * 0.01

    cv2.circle(frame, (lane_center, height - 10), 5, (255, 0, 0), -1)  # Lane center as a blue circle
    cv2.circle(frame, (car_position, height - 10), 5, (0, 0, 255), -1)  # Car position as a red circle
    
    cv2.putText(frame, f"Offset: {offset:.2f} px", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Steering Angle: {steering_angle:.2f} degrees", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return offset, steering_angle

def ransac_line_fit(points):
    if points:
        points = np.array(points).reshape(-1, 2)
        model, inliers = ransac(points, LineModelND, min_samples=2, residual_threshold=5, max_trials=500)
        line_x = np.array([points[inliers][:, 0].min(), points[inliers][:, 0].max()])
        line_y = model.predict_y(line_x)

        # Calculate slope and intercept
        slope = (line_y[1] - line_y[0]) / (line_x[1] - line_x[0]) if (line_x[1] - line_x[0]) != 0 else 0
        intercept = line_y[0] - slope * line_x[0]

        return line_x, line_y, slope, intercept
    return None, None, None, None

def update_line_history(line_x, line_y, history):
    if line_x is not None and line_y is not None:
        history.append((line_x, line_y))
        if len(history) > history_length:
            history.pop(0)
    if history:
        avg_x = np.mean([line[0] for line in history], axis=0)
        avg_y = np.mean([line[1] for line in history], axis=0)
        return avg_x, avg_y
    return None, None

def detect_lane_lines_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read the video file.")
        return
    height, width = frame.shape[:2]
    initialize_trapezoid_points(width, height)
    cv2.namedWindow("Lane Detection")
    cv2.setMouseCallback("Lane Detection", mouse_callback)

    frame_count = 0

    with open("lane_formulas.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Left Slope", "Left Intercept", "Right Slope", "Right Intercept"])

        while cap.isOpened():
            ret, frame = cap.read()
            frame_count += 1
            if not ret or frame_count % frame_skip != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            mask = np.zeros_like(edges)
            cv2.fillPoly(mask, [np.array(trapezoid_points, np.int32)], 255)
            masked_edges = cv2.bitwise_and(edges, mask)

            lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi / 180, threshold=40, minLineLength=50, maxLineGap=100)

            left_points, right_points = [], []
            if lines is not None:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                        if abs(slope) > 0.5:
                            if slope < 0 and x1 < width / 2 and x2 < width / 2:
                                left_points.append(((x1, y1), (x2, y2)))
                            elif slope > 0 and x1 > width / 2 and x2 > width / 2:
                                right_points.append(((x1, y1), (x2, y2)))

            left_line_x, left_line_y, left_slope, left_intercept = ransac_line_fit(left_points)
            right_line_x, right_line_y, right_slope, right_intercept = ransac_line_fit(right_points)

            roi_y_top = trapezoid_points[2][1]
            roi_y_bottom = trapezoid_points[0][1]
            left_line_x, left_line_y = extend_line_within_trapezoid(left_line_x, left_line_y, roi_y_top, roi_y_bottom, trapezoid_points)
            right_line_x, right_line_y = extend_line_within_trapezoid(right_line_x, right_line_y, roi_y_top, roi_y_bottom, trapezoid_points)

            # Smooth the lines using history
            smoothed_left_x, smoothed_left_y = update_line_history(left_line_x, left_line_y, left_line_history)
            smoothed_right_x, smoothed_right_y = update_line_history(right_line_x, right_line_y, right_line_history)

            # Save line formulas to CSV
            writer.writerow([frame_count, left_slope, left_intercept, right_slope, right_intercept])

            # Draw the trapezoid and corner markers
            cv2.polylines(frame, [np.array(trapezoid_points)], isClosed=True, color=(0, 0, 255), thickness=2)
            for point in trapezoid_points:
                cv2.circle(frame, point, 5, (0, 255, 255), -1)

            # Draw the smoothed lane lines
            if smoothed_left_x is not None and smoothed_left_y is not None:
                cv2.line(frame, (int(smoothed_left_x[0]), int(smoothed_left_y[0])), (int(smoothed_left_x[1]), int(smoothed_left_y[1])), (0, 255, 0), 5)
            if smoothed_right_x is not None and smoothed_right_y is not None:
                cv2.line(frame, (int(smoothed_right_x[0]), int(smoothed_right_y[0])), (int(smoothed_right_x[1]), int(smoothed_right_y[1])), (0, 255, 0), 5)

            # Calculate and display lane center and steering info
            calculate_lane_center_and_steering(frame, smoothed_left_x, smoothed_right_x)

            # Save frame for visualization
            if save_visualizations:
                cv2.imwrite(os.path.join(output_folder, f"frame_{frame_count}.png"), frame)

            cv2.imshow("Lane Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
detect_lane_lines_in_video("dashcam_video.mp4")
