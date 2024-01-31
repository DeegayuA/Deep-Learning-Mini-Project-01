import os
import cv2
import numpy as np

def sort_contours(cnts, method="left-to-right"):
    # Initialize the reverse flag and sort index
    reverse = False
    i = 0

    # Handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # Handle if we are sorting against the y-coordinate rather than the x-coordinate
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # Construct the list of bounding boxes and sort them from top to bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # Return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Noise reduction with Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Optional: apply morphological operations to close gaps between character parts
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closing

def crop_characters(image_path, margin=5):
    # Read the image
    image = cv2.imread(image_path)
    # Preprocess the image to get the binary image
    thresh = preprocess_image(image)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours
    sorted_contours, _ = sort_contours(contours, method="top-to-bottom")

    # Crop characters with margin
    cropped_characters = []
    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Add margin to each side of the character
        x_margin = max(0, x - margin)
        y_margin = max(0, y - margin)
        w_margin = min(image.shape[1], x + w + margin) - x_margin
        h_margin = min(image.shape[0], y + h + margin) - y_margin
        cropped = thresh[y_margin:y_margin + h_margin, x_margin:x_margin + w_margin]
        # Resize and normalize the cropped image
        cropped = cv2.resize(cropped, (28, 28)) # Resize to the desired size for your model
        cropped = cropped.astype(np.float32) / 255.0 # Normalize the pixel values
        cropped_characters.append(cropped)

        if len(cropped_characters) == 50: # Stop after cropping 50 characters
            break

    return cropped_characters

# Usage
image_path = 'img6.png'
characters = crop_characters(image_path, margin=5)

# Create a directory to save the cropped images
output_dir = 'cropped_characters'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save each cropped character with margin
for i, character in enumerate(characters):
    # Construct the filename using the format R1.jpg, R2.jpg, ..., R50.jpg
    filename = os.path.join(output_dir, f'P{i+1}.jpg')
    # Multiply by 255 as cv2.imwrite expects the image in the range [0, 255]
    character_to_save = (character * 255).astype(np.uint8)
    cv2.imwrite(filename, character_to_save)

print(f"All characters have been saved as separate JPG files inside the folder '{output_dir}'.")