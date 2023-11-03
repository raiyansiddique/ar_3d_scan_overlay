from PIL import Image

def image_to_8x8_matrix(image_path):
    # Load the image
    image = Image.open(image_path)

    # Get the size of the image
    image_width, image_height = image.size

    # Calculate the size of each square
    square_width = image_width // 8
    square_height = image_height // 8

    # Create an 8x8 matrix to hold the predominant color values
    matrix = [[None for _ in range(8)] for _ in range(8)]

    # Analyze each square to determine if its predominant color is white or black
    for i in range(8):
        for j in range(8):
            # Coordinates of the current square
            left = j * square_width
            upper = i * square_height
            right = left + square_width
            lower = upper + square_height
            
            # Crop the square from the image
            square = image.crop((left, upper, right, lower))
            
            # Calculate the average color of the square in grayscale
            avg_color = sum(square.convert("L").getdata()) / (square_width * square_height)
            
            # Determine if the square is predominantly white or black
            matrix[i][j] = 1 if avg_color > 127 else 0

    return matrix

def match_april_tag(tag):
    aprilTag = [[0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 0, 1, 0],
                [0, 0, 1, 1, 1, 0, 1, 0],
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]]

    # Function to rotate matrix 90 degrees clockwise
    def rotate_90(matrix):
        return [list(reversed(col)) for col in zip(*matrix)]

    # Check all rotations
    for _ in range(4):
        if tag == aprilTag:
            return True
        aprilTag = rotate_90(aprilTag)  # Rotate for next comparison

    # If none of the orientations match, return False
    return False



# Example usage:
# tag at 0 degrees
image_path = 'aprilTag2.png'
matrix0 = image_to_8x8_matrix(image_path)

# tag at 90 degrees
image_path = 'aprilTag2.png'
matrix90 = image_to_8x8_matrix(image_path)

# tag at 180 degrees
image_path = 'aprilTag2.png'
matrix180 = image_to_8x8_matrix(image_path)

# tag at 2700 degrees
image_path = 'aprilTag2.png'
matrix270 = image_to_8x8_matrix(image_path)

real_april_tag0 = match_april_tag(matrix0)
real_april_tag90 = match_april_tag(matrix90)
real_april_tag180 = match_april_tag(matrix180)
real_april_tag270 = match_april_tag(matrix270)

print(real_april_tag0)
print(real_april_tag90)
print(real_april_tag180)
print(real_april_tag270)


    