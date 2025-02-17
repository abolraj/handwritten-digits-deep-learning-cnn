import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('improved_trained_model.keras')

# Initialize drawing variables
drawing = False  # True if mouse is pressedq
ix, iy = -1, -1  # Initial mouse position

# Create a black canvas
canvas = np.zeros((280, 280, 1), dtype=np.uint8)  # 280x280 canvas (10x MNIST size)

# Mouse callback function
def draw(event, x, y, flags, param):
    global ix, iy, drawing, canvas

    if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:  # Draw while moving
        if drawing:
            cv2.line(canvas, (ix, iy), (x, y), (255, 255, 255), 15)  # Draw white line
            ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:  # Stop drawing
        drawing = False
        cv2.line(canvas, (ix, iy), (x, y), (255, 255, 255), 15)  # Final line

# Create a window and bind the mouse callback
cv2.namedWindow('Draw a Digit')
cv2.setMouseCallback('Draw a Digit', draw)

while True:
    # Display the canvas
    cv2.imshow('Draw a Digit', canvas)

    # Preprocess the drawn image for prediction
    resized = cv2.resize(canvas, (28, 28))  # Resize to 28x28 (MNIST input size)
    normalized = resized / 255.0  # Normalize pixel values
    input_data = np.expand_dims(normalized, axis=-1)  # Add channel dimension
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

    if drawing == False :
        
        # Predict the digit
        predictions = model.predict(input_data)
        predicted_digit = np.argmax(predictions)

        # Display the predicted digit
        print(f"Predicted Digit: {predicted_digit}")

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # Clear the canvas on 'c' key press
    if key == ord('c'):
        canvas = np.zeros((280, 280, 1), dtype=np.uint8)  # Reset canvas

    # Exit on 'q' key press
    elif key == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()