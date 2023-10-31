import cv2
import math

CONTOUR_PERMITER_THRESHOLD = 100
LINE_SMOOTHING_EPSILON = 0.01


# def reorder_vertices(contour):
#     vertices = [contour[0][0], contour[1][0], contour[2][0], contour[3][0]]

#     # Function to calculate distance from the origin
#     def calculate_distance_from_origin(point):
#         return math.sqrt(point[0] ** 2 + point[1] ** 2)

#     # Add each vertex's distance from the origin along with the vertex itself to a new list
#     vertices_with_distances = [
#         (vertex, calculate_distance_from_origin(vertex)) for vertex in vertices
#     ]

#     # Sort vertices based on their distances from the origin
#     sorted_vertices_with_distances = min(
#         vertices_with_distances, key=lambda item: item[1]
#     )

#     print(sorted_vertices_with_distances[0])


# Load an image
image = cv2.imread("image1.jpg")

# Convert it to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
ret, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

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
    if len(approximated_contour) == 4:
        # reorder_vertices(approximated_contour)
        cv2.drawContours(image, [approximated_contour], -1, (0, 255, 0), 2)

cv2.imshow("Image with quadrilaterals", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
