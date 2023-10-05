# Cone Detection and Lane Finding Using Computer Vision

This repository contains Python code that detects and highlights red traffic cones in an image while also finding the best-fit lines for the left and right sides of the detected cones. This project is particularly useful for applications in computer vision and robotics where the identification of objects like traffic cones is essential.

## My Approach

The algorithm employed in this project is inspired by [razimgit's project](https://gist.github.com/razimgit/d9c91edfd1be6420f58a74e1837bde18). It can be summarized in the following steps:

1. **Load and Prepare the Image**
   - Load an image and convert it to both RGB and HSV color spaces.

2. **Color Thresholding**
   - Define HSV color thresholds to isolate pixels with the approximate color of red cones.
   - Apply morphological operations to refine the thresholded image.
   - Blur the image slightly to further improve the quality of the thresholded image.

3. **Edge Detection and Contours**
   - Use the Canny edge detection algorithm to detect edges in the refined image.
   - Find contours in the edge-detected image.
   - Approximate and simplify the detected contours using the Douglas-Peucker algorithm.
   - Compute the convex hulls of the approximated contours.
   - Filter out contours with more than 10 or fewer than 3 points to focus on potential cone shapes.

4. **Identifying Upward-Pointing Cones**
   - Define a function to check if a convex hull represents an upward-pointing cone.
   - Determine the orientation of each identified convex hull based on bounding rectangles and filter out the ones pointing up.

5. **Highlighting Cones**
   - Draw rectangles around the identified cones, highlighting them on the original image.

6. **Lane Finding**
   - Fit best-fit lines to the left and right sides of the screen using the least-squares method.
   - Draw these best-fit lines on the image, representing the estimated lanes.

## Dependencies

- Python 3
- OpenCV (cv2)
- NumPy
- Matplotlib
- SciPy

## Usage

1. Clone this repository to your local machine.

2. Place your input image, named "red.png," in the same directory as the Python script.

3. Run the Python script:

   ```shell
   python cone_detection_and_lane_finding.py
