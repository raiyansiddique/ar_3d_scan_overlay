import cv2
import math
import numpy as np
import time

CODES_5X5 = {
    "0": np.array(
        [
            [255, 0, 255, 0, 0],
            [0, 255, 0, 255, 255],
            [0, 255, 255, 0, 0],
            [255, 0, 255, 0, 255],
            [255, 255, 255, 0, 0],
        ]
    ),
    "1": np.array(
        [
            [0, 0, 0, 0, 255],
            [255, 255, 0, 0, 0],
            [0, 0, 0, 0, 255],
            [255, 0, 255, 255, 255],
            [0, 0, 255, 255, 0],
        ]
    ),
}

CODES_4X4 = {
    "0": np.array(
        [[255, 0, 255, 255], [0, 255, 0, 255], [0, 0, 255, 255], [0, 0, 255, 0]]
    )
}

CONTOUR_PERIMETER_THRESHOLD = 100
LINE_SMOOTHING_EPSILON = 0.03
DST_IMG_SIZE = 128
APRIL_TAG_WIDTH = 6
IMAGE_PATH = "1.jpg"
MINIMUM_QUAD_AREA = 0.000001

PERIMETER_TOO_SMALL_CHECK = 1
IS_QUAD_CONCAVE_CHECK = 1
QUAD_AREA_TOO_SMALL_CHECK = 1


def create_dst_grid():
    # Calculate the spacing between points based on the desired grid size
    step_size = DST_IMG_SIZE / (APRIL_TAG_WIDTH)
    half_step_size = step_size / 2

    # Create a grid of points
    grid_points = []
    for i in range(APRIL_TAG_WIDTH):
        row = []
        for j in range(APRIL_TAG_WIDTH):
            # Compute the coordinates for each point
            x = j * step_size + half_step_size
            y = i * step_size + half_step_size
            row.append((x, y, 1))
        grid_points.append(row)

    return np.array(grid_points)


DST_GRID = create_dst_grid()


def is_quad_concave(vertices):
    # Ensure the vertices are ordered consistently (e.g., clockwise or counterclockwise)
    # vertices should be a list of points, e.g., [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    n = len(vertices)
    sign = 0
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        p3 = vertices[(i + 2) % n]

        # Compute vectors for the edges: v1 from p1 to p2, v2 from p2 to p3
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p2)

        # Calculate the Z component of the cross product (v1 x v2)
        cross_product_z = np.cross(v1, v2)
        # If the sign changes, we have a concave corner
        if i == 0:  # First corner, just record the sign
            sign = np.sign(cross_product_z)
        else:
            if np.sign(cross_product_z) != sign:
                return True  # Concave corner detected

    return False  # No concave corners detected


def perimeter_too_small(contour):
    # Calculate the arc length (perimeter) of your contour
    contour_perimeter = cv2.arcLength(contour, True)

    # Filter out contours that are too small
    if contour_perimeter < CONTOUR_PERIMETER_THRESHOLD:
        return True
    return False


def quad_too_small(corners):
    # Find area of detected quadrilateral
    quad_area = find_quad_area(corners)

    # Make area proportional to the image size
    quad_area_proportion = quad_area / (IMAGE_WIDTH * IMAGE_HEIGHT)

    # Filter out quadrilaterals that are too small
    if quad_area_proportion < MINIMUM_QUAD_AREA:
        return True
    return False


def find_quad_area(corners):
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


def match_code(code):
    for key in CODES_4X4:
        for i in range(4):
            if np.array_equal(code, np.rot90(CODES_4X4[key], i)):
                return key
    return None


def apply_homography_to_grid(grid_points, homography):
    # Flatten the grid to a (N, 3) matrix where N is grid_size^2
    flat_grid_points = grid_points.reshape(-1, 3).T

    # Apply homography matrix to all points using matrix multiplication
    # The result will be in homogeneous coordinates
    transformed_points_homogeneous = homography @ flat_grid_points

    # Normalize points to convert from homogeneous to Cartesian coordinates
    transformed_points_cartesian = (
        transformed_points_homogeneous[:2, :] / transformed_points_homogeneous[2, :]
    )

    # Reshape the points back to the original grid shape
    transformed_grid = transformed_points_cartesian.T.reshape(
        grid_points.shape[0], grid_points.shape[1], 2
    )

    # Append a third column of ones to represent the homogeneous coordinate
    ones = np.ones((grid_points.shape[0], grid_points.shape[1], 1))
    transformed_grid = np.concatenate([transformed_grid, ones], axis=2)

    return transformed_grid


def extract_pixel_values(image, coords):
    # Ensure the image is grayscale
    if len(image.shape) > 2:
        raise ValueError("Input image must be grayscale")

    # Ensure the coordinates are within the image bounds
    coords = np.clip(coords, 0, np.array(image.shape[:2])[::-1] - 1)

    # Round the coordinates to the nearest integer to use them as indices
    coords = np.rint(coords).astype(int)

    # Extract the pixel values from the image
    pixel_values = image[coords[..., 1], coords[..., 0]]

    return pixel_values


def homography(pts_src):
    # Assuming pts_src is correctly provided as the four corners of the quad
    # and pts_dst is the target square's corners.
    pts_dst = np.array(
        [
            [0, 0],
            [DST_IMG_SIZE - 1, 0],
            [DST_IMG_SIZE - 1, DST_IMG_SIZE - 1],
            [0, DST_IMG_SIZE - 1],
        ],
        dtype="float32",
    )

    # Calculate the inverse homography matrix from the destination (square) to the source (quad)
    h_inv, _ = cv2.findHomography(pts_dst, pts_src)
    return h_inv


# Instead of loading an image, start video capture
cap = cv2.VideoCapture(0)  # '0' is typically the default value for the primary camera

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process frames in a loop
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from video stream.")
        break

    start_time = time.time()

    # Your image processing code starts here
    # Replace 'image' with 'frame' and remove the line that reads the image from a file
    IMAGE_WIDTH = frame.shape[1]
    IMAGE_HEIGHT = frame.shape[0]

    # Convert it to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get the average value
    average_brightness = cv2.mean(gray_image)[0]

    # Apply binary thresholding
    ret, binary_image = cv2.threshold(
        gray_image, average_brightness, 255, cv2.THRESH_BINARY
    )

    # Find contours
    contours, hierarchy = cv2.findContours(
        binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        # Calculate the arc length (perimeter) of your contour
        contour_perimeter = cv2.arcLength(contour, True)

        if perimeter_too_small(contour) and PERIMETER_TOO_SMALL_CHECK:
            continue

        # Use the arc length to help determine the 'epsilon' parameter
        epsilon = LINE_SMOOTHING_EPSILON * contour_perimeter

        # Use approxPolyDP to simplify your contour
        approximated_contour = cv2.approxPolyDP(contour, epsilon, True)

        if len(approximated_contour) == 4:
            vertices = np.array(
                [
                    [
                        approximated_contour[0][0][0],
                        approximated_contour[0][0][1],
                    ],
                    [
                        approximated_contour[1][0][0],
                        approximated_contour[1][0][1],
                    ],
                    [
                        approximated_contour[2][0][0],
                        approximated_contour[2][0][1],
                    ],
                    [
                        approximated_contour[3][0][0],
                        approximated_contour[3][0][1],
                    ],
                ]
            )

            if is_quad_concave(vertices) and IS_QUAD_CONCAVE_CHECK:
                continue

            # Find the homography matrix from a square to the detected quadrilateral
            dst_to_src_homography = homography(vertices)

            # Find the marker in the binary image using the homography
            # marker_matrix = get_marker_matrix(binary_image, dst_to_src_homography)
            marker_matrix_coords = apply_homography_to_grid(
                DST_GRID, dst_to_src_homography
            )
            marker_matrix = extract_pixel_values(
                binary_image, marker_matrix_coords[..., :2]
            )

            marker_code = match_code(marker_matrix[1:-1, 1:-1])

            # Draw the marker code on the image
            if marker_code is not None:
                # Calculate the center of the quad
                center = np.mean(vertices, axis=0).astype(int)

                # Draw the marker code on the image
                cv2.putText(
                    frame,
                    str(marker_code),
                    tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.drawContours(frame, [approximated_contour], -1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Image with quadrilaterals", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if cv2.waitKey(1) & 0xFF == ord("i"):
        if PERIMETER_TOO_SMALL_CHECK:
            PERIMETER_TOO_SMALL_CHECK = False
            print("PERIMETER_TOO_SMALL_CHECK: False")
        else:
            PERIMETER_TOO_SMALL_CHECK = True
            print("PERIMETER_TOO_SMALL_CHECK: True")

    if cv2.waitKey(1) & 0xFF == ord("o"):
        if IS_QUAD_CONCAVE_CHECK:
            IS_QUAD_CONCAVE_CHECK = False
            print("IS_QUAD_CONCAVE_CHECK: False")
        else:
            IS_QUAD_CONCAVE_CHECK = True
            print("IS_QUAD_CONCAVE_CHECK: True")

    if cv2.waitKey(1) & 0xFF == ord("p"):
        if QUAD_AREA_TOO_SMALL_CHECK:
            QUAD_AREA_TOO_SMALL_CHECK = False
            print("QUAD_AREA_TOO_SMALL_CHECK: False")
        else:
            QUAD_AREA_TOO_SMALL_CHECK = True
            print("QUAD_AREA_TOO_SMALL_CHECK: True")

    # Convert it to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get the average value
    average_brightness = cv2.mean(gray_image)[0]

    # Apply

    # Calculate and display FPS
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"FPS: {elapsed_time:.2f}")

# When everything done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
