import cv2
import math
import numpy as np

CONTOUR_PERMITER_THRESHOLD = 100
LINE_SMOOTHING_EPSILON = 0.03
SQUARE_IMAGE_RESOLUTION = 600
APRIL_TAG_WIDTH = 8
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


def homography(img_src, pts_src):
    img_src = cv2.imread(img_src)

    pts_dst = np.array(
        [
            [0, 0],
            [APRIL_TAG_WIDTH - 1, 0],
            [APRIL_TAG_WIDTH - 1, APRIL_TAG_WIDTH - 1],
            [0, APRIL_TAG_WIDTH - 1],
        ]
    )

    h, status = cv2.findHomography(pts_src, pts_dst)
    im_out = cv2.warpPerspective(img_src, h, (APRIL_TAG_WIDTH, APRIL_TAG_WIDTH))
    im_out = im_out[1:-1, 1:-1]
    cv2.imshow("Warped Source Image", im_out)

    cv2.waitKey(0)


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
        vertices = np.array(
            [
                [
                    approximated_contour[0][0][0] / IMAGE_WIDTH,
                    approximated_contour[0][0][1] / IMAGE_HEIGHT,
                ],
                [
                    approximated_contour[1][0][0] / IMAGE_WIDTH,
                    approximated_contour[1][0][1] / IMAGE_HEIGHT,
                ],
                [
                    approximated_contour[2][0][0] / IMAGE_WIDTH,
                    approximated_contour[2][0][1] / IMAGE_HEIGHT,
                ],
                [
                    approximated_contour[3][0][0] / IMAGE_WIDTH,
                    approximated_contour[3][0][1] / IMAGE_HEIGHT,
                ],
            ]
        )
        quad_area = find_quad_area(vertices)
        print(quad_area)
        if quad_area < 0.01:
            continue
        # homography(IMAGE_PATH, vertices)
        cv2.drawContours(image, [approximated_contour], -1, (0, 255, 0), 2)

cv2.imshow("Image with quadrilaterals", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
