import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
num_points = 1000  # Number of points on the cube surface
cube_side = 2  # Length of the cube sides

# Initialize points arrays
points = np.zeros((num_points, 3))

# Randomly populate the points on the surface of the cube
for i in range(num_points):
    # Randomly select a face (each face has equal probability)
    face = np.random.choice(["top", "bottom", "front", "back", "left", "right"])

    if face == "top":
        points[i] = [
            np.random.rand() * cube_side,
            cube_side,
            np.random.rand() * cube_side,
        ]
    elif face == "bottom":
        points[i] = [np.random.rand() * cube_side, 0, np.random.rand() * cube_side]
    elif face == "front":
        points[i] = [np.random.rand() * cube_side, np.random.rand() * cube_side, 0]
    elif face == "back":
        points[i] = [
            np.random.rand() * cube_side,
            np.random.rand() * cube_side,
            cube_side,
        ]
    elif face == "left":
        points[i] = [0, np.random.rand() * cube_side, np.random.rand() * cube_side]
    elif face == "right":
        points[i] = [
            cube_side,
            np.random.rand() * cube_side,
            np.random.rand() * cube_side,
        ]

# Visualization
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(points[:, 0], points[:, 1], points[:, 2])

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.title("3D Point Cloud of a Cube")
# plt.show()


def save_to_pcd_file(points, filename="cube.pcd"):
    """
    Save the 3D point cloud to a PCD file.

    :param points: numpy array of points.
    :param filename: path to file to save.
    """
    # PCD file header format
    pcd_header = """\
    VERSION .7
    FIELDS x y z
    SIZE 4 4 4
    TYPE F F F
    COUNT 1 1 1
    WIDTH {0}
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS {0}
    DATA ascii
    """

    # Number of points
    num_points = points.shape[0]

    # Open file in write mode
    with open(filename, "w") as file:
        # Write header to file
        file.write(pcd_header.format(num_points))

        # Write point cloud data to file
        for i in range(num_points):
            file.write("{0} {1} {2}\n".format(points[i, 0], points[i, 1], points[i, 2]))


# Assuming 'points' contains your 3D point cloud data from the previous script.
# Save point cloud to 'cube.pcd'
save_to_pcd_file(points, "cube2.pcd")
