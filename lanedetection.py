import cv2
import numpy as np
from skimage.measure import LineModelND, ransac


def img_show(image):
    cv2.imshow("Lane Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def detect_lane_lines(image_path):
    # Load the dashcam image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #img_show(gray)
    
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #img_show(blurred)
    
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    #img_show(edges)
    
    # Mask the edges image to focus on a larger region of interest
    height, width = edges.shape
    mask = np.zeros_like(edges)
    
    # Adjusted polygon to cover more of the image area
    polygon = np.array([[ 
    (0, height), (width, height), (int(width * 0.7), int(height * 0.6)), (int(width * 0.3), int(height * 0.6))
]], np.int32)

    
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Find line segments in the edges image with extended line length and gap tolerance
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi / 180, threshold=30, minLineLength=70, maxLineGap=80)

    print(lines)
    # Prepare points for RANSAC
    left_points, right_points = [], []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            if abs(slope) > 0.5:  # Filter for steep slopes
                if slope < 0:
                    left_points.extend([(x1, y1), (x2, y2)])
                else:
                    right_points.extend([(x1, y1), (x2, y2)])

    # Apply RANSAC to find the best line fit for left and right lane lines
    def ransac_line_fit(points):
        if points:
            points = np.array(points)
            model, inliers = ransac(points, LineModelND, min_samples=2, residual_threshold=2, max_trials=1000)
            line_x = np.array([points[inliers][:, 0].min(), points[inliers][:, 0].max()])
            line_y = model.predict_y(line_x)
            formula = (model.params[0], model.params[1])  # Slope and intercept
            return line_x, line_y, formula
        return None, None, None

    left_line_x, left_line_y, left_formula = ransac_line_fit(left_points)
    right_line_x, right_line_y, right_formula = ransac_line_fit(right_points)

    # Draw the detected lane lines
    if left_line_x is not None and left_line_y is not None:
        cv2.line(image, (int(left_line_x[0]), int(left_line_y[0])), (int(left_line_x[1]), int(left_line_y[1])), (0, 255, 0), 5)
    if right_line_x is not None and right_line_y is not None:
        cv2.line(image, (int(right_line_x[0]), int(right_line_y[0])), (int(right_line_x[1]), int(right_line_y[1])), (0, 255, 0), 5)

    # Display results
    cv2.imshow("Lane Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Output lane line formulas
    print(f"Left lane line formula (y = mx + b): y = {left_formula[0]}x + {left_formula[1]}" if left_formula else "No left lane line detected")
    print(f"Right lane line formula (y = mx + b): y = {right_formula[0]}x + {right_formula[1]}" if right_formula else "No right lane line detected")

# Example usage
detect_lane_lines("dashcam_image.jpg")
