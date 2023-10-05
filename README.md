# Cone Detection and Lane Finding Using Computer Vision

This repository contains Python code that detects and highlights red traffic cones in an image while also finding the best-fit lines for the left and right sides of the detected cones. This project is particularly useful for applications in computer vision and robotics where the identification of objects like traffic cones is essential.

## Methodology

The algorithm employed in this project is inspired by [razimgit's project](https://gist.github.com/razimgit/d9c91edfd1be6420f58a74e1837bde18). It can be summarized in the following steps:

### 1. Load and Prepare the Image

We begin by loading an image and converting it to both RGB and HSV color spaces. This step is crucial as it forms the basis for further analysis.

```
import cv2 as cv

# Opening image
img = cv.imread("red.png")

# Convert the image to RGB and HSV color spaces
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
```

### 2. Color Thresholding

To identify red traffic cones, we define HSV color thresholds that isolate pixels with the approximate color of red cones. We then apply morphological operations to refine the thresholded image and improve its quality.

```
# Define color thresholds
img_thresh_low = cv.inRange(img_hsv, np.array([0, 135, 135]), np.array([15, 255, 255]))
img_thresh_high = cv.inRange(img_hsv, np.array([159, 135, 135]), np.array([179, 255, 255]))

# Combine the two threshold maps
img_thresh = cv.bitwise_or(img_thresh_low, img_thresh_high)

# Apply morphological operations for noise removal
kernel = np.ones((5, 5))
img_thresh_opened = cv.morphologyEx(img_thresh, cv.MORPH_OPEN, kernel)

# Blur the image slightly for further refinement
img_thresh_blurred = cv.medianBlur(img_thresh_opened, 5)
```

### 3. Edge Detection and Contours

We employ the Canny edge detection algorithm to detect edges in the refined image. Subsequently, we find contours in the edge-detected image and simplify them using the Douglas-Peucker algorithm.

```
# Use Canny edge detection to find edges
img_edges = cv.Canny(img_thresh_blurred, 70, 255)

# Find contours in the edge-detected image
contours, _ = cv.findContours(np.array(img_edges), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Approximate and simplify the detected contours
approx_contours = []
for c in contours:
    approx = cv.approxPolyDP(c, 5, closed=True)
    approx_contours.append(approx)
```

### 4. Identifying Upward-Pointing Cones

We have a function that checks if a convex hull represents an upward-pointing cone based on bounding rectangles. This function filters out the upward-pointing cones from the rest.

```
def convex_hull_pointing_up(ch: np.ndarray) -> bool:
    # Define points above and below the vertical center line
    points_above_center, points_below_center = [], []
    _, y, _, h = cv.boundingRect(ch) 
    vertical_center = y + h / 2

    # Separate points based on their y-coordinates
    for point in ch:
        if point[0][1] < vertical_center: 
            points_above_center.append(point)
        elif point[0][1] >= vertical_center:
            points_below_center.append(point)

    # Determine the horizontal positions
    x_above, _, w_above, _ = cv.boundingRect(np.array(points_above_center)) 
    x_below, _, w_below, _ = cv.boundingRect(np.array(points_below_center))

    # Check if the cone points upward based on positions
    return x_above <= x_below + w_below and x_above + w_above <= x_below + w_below \
        and x_above >= x_below and x_above + w_above >= x_below
```

### 5. Highlighting Cones

We draw rectangles around the identified cones, effectively highlighting them on the original image.

```
# Highlight identified cones by drawing rectangles
for rect in bounding_rects:
    x, y, w, h = rect
    cv.rectangle(img_res, (x, y), (x + w, y + h), (0, 255, 0), 2)
```

### 6. Lane Finding

To estimate the lanes, we fit best-fit lines to the left and right sides of the screen using the least-squares method.

```
# Fit best-fit lines to the left and right sides of the screen
cone_points_left = [(rect[0] + rect[2] / 2, rect[1] + rect[3] / 2) for rect in bounding_rects if rect[0] + rect[2] / 2 < img_res.shape[1] / 2]
cone_points_right = [(rect[0] + rect[2] / 2, rect[1] + rect[3] / 2) for rect in bounding_rects if rect[0] + rect[2] / 2 > img_res.shape[1] / 2]

# Define a function for least squares fitting
def least_squares(x, y):
    def func(x, a, b):
        return a * x + b
    
    popt, pcov = optimize.curve_fit(func, x, y)
    return popt

# Get best-fit lines for left and right cones
a1, b1 = least_squares(np.array([i[0] for i in cone_points_left]), np.array([i[1] for i in cone_points_left]))
a2, b2 = least_squares(np.array([i[0] for i in cone_points_right]), np.array([i[1] for i in cone_points_right]))

# Draw the best-fit lines on the image
cv.line(img_res, [0, int(b1)], [3000, int((3000 * a1) + b1)], (255, 1, 1), 5)
cv.line(img_res, [0, int(b2)], [3000, int((3000 * a2) + b2)], (255, 1, 1), 5)
```

## Dependencies

- Python 3 (Visual Studio Code)
- OpenCV (cv2)
- NumPy (np)
- Matplotlib (pyplot)
- SciPy (SciPy.optimize)

