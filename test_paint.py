import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('improved_trained_model.keras')

# Initialize drawing variables
drawing = False
ix, iy = -1, -1

# Create a black canvas
canvas = np.zeros((280, 280), dtype=np.uint8)  # 280x280 canvas

# Mouse callback function
def draw(event, x, y, flags, param):
    global ix, iy, drawing, canvas

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(canvas, (ix, iy), (x, y), 255, 15)  # Draw white line
            ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(canvas, (ix, iy), (x, y), 255, 15)

# Create a window and bind the mouse callback
cv2.namedWindow('Draw a Digit')
cv2.setMouseCallback('Draw a Digit', draw)

while True:
    # Make a copy of the canvas to display text without altering the original drawing
    display_canvas = canvas.copy()

    if not drawing:
        # Preprocess the drawn image for prediction
        resized = cv2.resize(canvas, (28, 28))
        normalized = resized / 255.0
        input_data = normalized.reshape(1, 28, 28, 1)

        # Predict the digit
        predictions = model.predict(input_data)
        predicted_digit = np.argmax(predictions)

        # Display the predicted digit on the canvas
        text = f"Predicted Digit: {predicted_digit}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_position = (10, 40)
        font_scale = .5
        color = 255  # White color in grayscale
        thickness = 2

        # Put text on the display canvas
        cv2.putText(display_canvas, text, text_position, font, font_scale, color, thickness, cv2.LINE_AA)

    # Display the canvas
    cv2.imshow('Draw a Digit', display_canvas)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # Clear the canvas on 'c' key press
    if key == ord('c'):
        canvas = np.zeros((280, 280), dtype=np.uint8)

    # Exit on 'q' key press
    elif key == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
