import cv2
import math
import numpy as np

codes = {
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

CONTOUR_PERMITER_THRESHOLD = 100
LINE_SMOOTHING_EPSILON = 0.03
SQUARE_IMAGE_RESOLUTION = 600
APRIL_TAG_SIZE = 128
GRID_SIZE = 8
IMAGE_PATH = "markers.jpg"


def find_quad_area(vertices):
    line_lengths = []
    for i, vertex in enumerate(vertices):
        if i == len(vertices) - 1:
            next_vertex = vertices[0]
        else:
            next_vertex = vertices[i + 1]
        line_lengths.append(
            math.sqrt(
                abs(vertex[0] - next_vertex[0]) ** 2
                + abs(next_vertex[1] - next_vertex[1]) ** 2
            )
        )
    quad_area = 0.5 * (line_lengths[0] + line_lengths[1]) + 0.5 * (
        line_lengths[2] + line_lengths[3]
    )
    return quad_area


def match_code(code):
    for key in codes:
        for i in range(4):
            if np.array_equal(code, np.rot90(codes[key], i)):
                return key
    return None


def get_marker_matrix(binary_image, homography):
    # We need to calculate the step size for the grid, which is the size of the tag divided by 7
    step_size = APRIL_TAG_SIZE / GRID_SIZE
    # step_size = 1
    half_step_size = step_size / 2

    # Sample points from the grid
    sampled_points = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            # Create a homogeneous coordinate for the point in the destination image
            # We multiply the step size by i or j to get the actual coordinate
            dst_point = np.array(
                [i * step_size + half_step_size, j * step_size + half_step_size, 1]
            )

            # Use the inverse homography matrix to transform the point to the source space
            src_point_homogeneous = np.dot(homography, dst_point)

            # Convert back to 2D by normalizing with respect to the third coordinate
            src_point = src_point_homogeneous / src_point_homogeneous[2]

            # The coordinates in the source image (quad space)
            x_src, y_src = src_point[0], src_point[1]

            # Add the source points to the list as integer tuples
            pixel_value = binary_image[int(round(y_src)), int(round(x_src))]
            sampled_points.append(pixel_value)

    # Reshape the sampled_points list to 8 by 8
    sampled_points = np.reshape(sampled_points, (8, 8))
    sampled_points = np.flip(sampled_points, 0)
    sampled_points = sampled_points[1:-1, 1:-1]
    print(sampled_points)
    # print(sampled_points)
    # cv2.imshow("Image with quadrilaterals", sampled_points)
    # cv2.imwrite("warped.png", sampled_points)
    # cv2.waitKey(0)
    return sampled_points


def homography(pts_src):
    # Assuming pts_src is correctly provided as the four corners of the quad
    # and pts_dst is the target square's corners.
    pts_dst = np.array(
        [
            [0, 0],
            [APRIL_TAG_SIZE - 1, 0],
            [APRIL_TAG_SIZE - 1, APRIL_TAG_SIZE - 1],
            [0, APRIL_TAG_SIZE - 1],
        ],
        dtype="float32",
    )

    # Calculate the inverse homography matrix from the destination (square) to the source (quad)
    h_inv, status = cv2.findHomography(pts_dst, pts_src)
    return h_inv


# def image_to_8x8_matrix(image_path):

# Load an image
image = cv2.imread(IMAGE_PATH)

IMAGE_WIDTH = image.shape[1]
IMAGE_HEIGHT = image.shape[0]

# Convert it to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
    if contour_perimeter < CONTOUR_PERMITER_THRESHOLD:
        continue

    # Use the arc length to help determine the 'epsilon' parameter
    epsilon = LINE_SMOOTHING_EPSILON * contour_perimeter

    # Use approxPolyDP to simplify your contour
    approximated_contour = cv2.approxPolyDP(contour, epsilon, True)
    # cv2.drawContours(image, [approximated_contour], -1, (0, 255, 0), 2)

    if len(approximated_contour) == 4:
        # vertices = np.array(
        #     [
        #         [
        #             approximated_contour[0][0][0] / IMAGE_WIDTH,
        #             approximated_contour[0][0][1] / IMAGE_HEIGHT,
        #         ],
        #         [
        #             approximated_contour[1][0][0] / IMAGE_WIDTH,
        #             approximated_contour[1][0][1] / IMAGE_HEIGHT,
        #         ],
        #         [
        #             approximated_contour[2][0][0] / IMAGE_WIDTH,
        #             approximated_contour[2][0][1] / IMAGE_HEIGHT,
        #         ],
        #         [
        #             approximated_contour[3][0][0] / IMAGE_WIDTH,
        #             approximated_contour[3][0][1] / IMAGE_HEIGHT,
        #         ],
        #     ]
        # )
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
        quad_area = find_quad_area(vertices)
        # center_of_mass = find_center_of_mass(vertices)
        # print(quad_area)
        if quad_area > 100:
            continue
        dst_to_src_homography = homography(vertices)
        marker_matrix = get_marker_matrix(binary_image, dst_to_src_homography)
        print(match_code(marker_matrix))
        # cv2.drawContours(image, [approximated_contour], -1, (0, 255, 0), 2)

# cv2.imshow("Image with quadrilaterals", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
