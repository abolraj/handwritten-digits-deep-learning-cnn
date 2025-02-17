import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('improved_trained_model.keras')

# Start capturing video
cap = cv2.VideoCapture(0)

# Coordinates for ROI
x0, y0, width, height = 300, 100, 200, 200

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optionally flip the frame horizontally if needed
    # frame = cv2.flip(frame, 1)

    roi = frame[y0:y0 + height, x0:x0 + width]
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 2)

    # Preprocessing
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        contour = max(contours, key=cv2.contourArea)

        # Proceed only if the contour is large enough
        if cv2.contourArea(contour) > 100:
            # Get bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(contour)
            padding = 10

            a = max(w,h)

            # Adjust coordinates to include padding and stay within image bounds
            x_start = max(x - padding - (a - w)//2, 0)
            y_start = max(y - padding - (a - h)//2, 0)
            x_end = min(x + a + padding * 2, thresh.shape[1])
            y_end = min(y + a + padding * 2, thresh.shape[0])

            # Draw the bounding rectangle around the digit
            cv2.rectangle(roi, (x_start, y_start), (x_end, y_end), (50, 50, 200), 2)

            # Crop the digit from the thresholded image
            digit = thresh[y_start:y_end, x_start:x_end]

            # Resize the cropped image to 28x28 pixels
            resized = cv2.resize(digit, (28, 28))
            normalized = resized / 255.0
            input_data = normalized.reshape(1, 28, 28, 1)

            # Predict the digit
            prediction = model.predict(input_data)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)

            # Display the prediction on the frame
            cv2.putText(frame, f"Digit: {predicted_digit} ({confidence * 100:.2f}%)",
                        (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "No digit detected", (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No digit detected", (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with detection
    cv2.imshow("Digit Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
