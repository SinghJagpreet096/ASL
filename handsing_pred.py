from tensorflow import keras
import cv2
import numpy as np


def load_model():
    return keras.models.load_model('signWageM1.h5',compile=False)


model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to 28x28 pixels
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    # Reshape the image to match the model input shape
    reshaped = resized.reshape(1, 28, 28, 1)
    # Normalize the image
    normalized = reshaped.astype('float32') / 255
    return normalized

# Function to predict the digit
def predict_digit(image):
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed)
    digit = np.argmax(prediction)
    return digit

# Capture image from webcam
cap = cv2.VideoCapture(1)

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Detect key press
    key = cv2.waitKey(1)
    if key == ord('c'):
        # Capture the frame
        captured_image = frame.copy()

        # Preprocess and predict the digit
        digit = predict_digit(captured_image)

        # Display the predicted digit
        cv2.putText(captured_image, f"Digit: {digit}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Captured Image', captured_image)

    elif key == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()