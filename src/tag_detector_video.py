import cv2
import math
import numpy as np
import time

# Matrix representation of the 4x4 AprilTag codes
CODES_4X4 = {
    "0": np.array(
        [[255, 0, 255, 255], [0, 255, 0, 255], [0, 0, 255, 255], [0, 0, 255, 0]]
    ),
    "1": np.array(
        [[0, 0, 0, 0, 0], [255, 255, 255, 255], [255, 0, 0, 255], [255, 0, 255, 0]]
    ),
    "2": np.array(
        [[0, 0, 255, 255], [0, 0, 255, 255], [0, 0, 255, 0], [255, 255, 0, 255]]
    ),
    "3": np.array(
        [[255, 0, 0, 255], [255, 0, 0, 255], [0, 255, 0, 0], [0, 255, 255, 0]]
    ),
    "4": np.array(
        [[0, 255, 0, 255], [0, 255, 0, 0], [255, 0, 0, 255], [255, 255, 255, 0]]
    ),
}

# Thresholds for rejecting unfit quadrilaterals
CONTOUR_PERIMETER_THRESHOLD = 100
MINIMUM_QUAD_AREA = 0.000001
LINE_SMOOTHING_EPSILON = 0.03

# Bigger image size means more accurate homography
DST_IMG_SIZE = 128
APRIL_TAG_WIDTH = 6

# Toggle checks on quadrilateral
PERIMETER_TOO_SMALL_CHECK = 1
IS_QUAD_CONCAVE_CHECK = 1
QUAD_AREA_TOO_SMALL_CHECK = 1


def create_square_grid():
    """
    Creates a grid of points for the square image based on the desired grid size and AprilTag width.

    Returns:
    np.array: A numpy array of shape (APRIL_TAG_WIDTH, APRIL_TAG_WIDTH, 3) containing the coordinates of each point in the grid.
    """

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


DST_GRID = create_square_grid()


def is_quad_concave(vertices):
    """
    Determines whether a quadrilateral defined by the given vertices is concave.

    Args:
        vertices (list): A list of four points, each represented as a tuple of (x, y) coordinates.

    Returns:
        bool: True if the quadrilateral is concave, False otherwise.
    """
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


def quad_too_small(corners):
    """
    Check if the detected quadrilateral is too small.

    Args:
        corners (list): List of four corner points of the quadrilateral.

    Returns:
        bool: True if the quadrilateral is too small, False otherwise.
    """
    # Find area of detected quadrilateral
    quad_area = find_quad_area(corners)

    # Make area proportional to the image size
    image_area = IMAGE_WIDTH * IMAGE_HEIGHT
    quad_area_proportional = quad_area / image_area

    # Filter out quadrilaterals that are too small
    if quad_area_proportional < MINIMUM_QUAD_AREA:
        return True
    return False


def find_quad_area(corners):
    """
    Calculates the area of a quadrilateral given its four corners.

    Args:
        corners (list of tuples): The (x, y) coordinates of the four corners of the quadrilateral.

    Returns:
        float: The area of the quadrilateral.
    """
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


def homography(quad_corners):
    """
    Calculates the inverse homography matrix from a set of corners of a square to a set of corners of a quadrilateral.

    Args:
        quad_corners (numpy.ndarray): An array of shape (4, 2) containing the (x, y) coordinates of the square corners.

    Returns:
        numpy.ndarray: An array of shape (3, 3) representing the inverse homography matrix.
    """
    square_corners = np.array(
        [
            [0, 0],
            [DST_IMG_SIZE - 1, 0],
            [DST_IMG_SIZE - 1, DST_IMG_SIZE - 1],
            [0, DST_IMG_SIZE - 1],
        ],
        dtype="float32",
    )

    # Calculate the inverse homography matrix from the destination (square) to the source (quad)
    homography, _ = cv2.findHomography(square_corners, quad_corners)
    return homography


def apply_homography_to_grid(grid_points, homography):
    """
    Applies a homography matrix to a grid of points.

    Args:
        grid_points (numpy.ndarray): A grid of points in 3D space, with shape (rows, cols, 3).
        homography (numpy.ndarray): A 3x3 homography matrix.

    Returns:
        numpy.ndarray: The transformed grid of points in 2D space, with shape (rows, cols, 3).
    """

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
    # ones = np.ones((grid_points.shape[0], grid_points.shape[1], 1))
    # transformed_grid = np.concatenate([transformed_grid, ones], axis=2)

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


def match_code(code):
    """
    Matches a given 4x4 april tag matrix with the dictionary of known codes.

    Args:
        code (numpy.ndarray): A 4x4 numpy array representing the code to be matched.

    Returns:
        str: The key of the matched code in the CODES_4X4 dictionary, or None if no match is found.
    """
    for key in CODES_4X4:
        for i in range(4):
            if np.array_equal(code, np.rot90(CODES_4X4[key], i)):
                return key
    return None


if __name__ == "__main__":
    # Start video capture
    cap = cv2.VideoCapture(0)

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

        IMAGE_WIDTH = frame.shape[1]
        IMAGE_HEIGHT = frame.shape[0]

        # Convert it to grayscale
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get the average brightness value of pixels in the frame.
        # This is our threshold for binary thresholding.
        average_brightness = cv2.mean(gray_image)[0]

        # Apply binary thresholding to the frame
        ret, binary_image = cv2.threshold(
            gray_image, average_brightness, 255, cv2.THRESH_BINARY
        )

        # Find contours in frame
        contours, hierarchy = cv2.findContours(
            binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            # Calculate the arc length (perimeter) of the contour
            contour_perimeter = cv2.arcLength(contour, True)

            # Reject contours that are too small
            if (
                contour_perimeter < CONTOUR_PERIMETER_THRESHOLD
                and PERIMETER_TOO_SMALL_CHECK
            ):
                continue

            # Use the arc length to help determine the 'epsilon' parameter
            # which determines how much the contour is simplified
            epsilon = LINE_SMOOTHING_EPSILON * contour_perimeter

            # Use approxPolyDP to simplify the contour
            approximated_contour = cv2.approxPolyDP(contour, epsilon, True)

            # If the contour is not a quadrilateral, skip it
            if len(approximated_contour) != 4:
                continue

            # Get the coordinates of the four corners of the quadrilateral.
            # These are guaranteed to be in consecutive order, but clockwise
            # or counterclockwise is not guaranteed.
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

            # Reject concave quadrilaterals
            if is_quad_concave(vertices) and IS_QUAD_CONCAVE_CHECK:
                continue

            # Find the homography matrix from a square to the detected quadrilateral.
            # This homography helps us sample points in the quadrilateral in a grid like
            # fashion.
            square_to_quad_homography = homography(vertices)

            # Returns the corresponding coordinates of a grid of points from a square to
            # the detected quadrilateral.
            marker_matrix_coords = apply_homography_to_grid(
                DST_GRID, square_to_quad_homography
            )

            # Extract the pixel values from the binary image at the marker matrix coordinates
            marker_matrix = extract_pixel_values(binary_image, marker_matrix_coords)

            # We get rid of the border pixels of the marker since they are always black.
            # Match the
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

                # Draw the detected quadrilateral on the image
                cv2.drawContours(frame, [approximated_contour], -1, (0, 255, 0), 2)

        # Calculate and display time to process
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Delay: {elapsed_time:.5f} seconds")

        # Display the resulting frame
        cv2.imshow("Livefeed", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Toggle checks on quadrilateral
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

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
