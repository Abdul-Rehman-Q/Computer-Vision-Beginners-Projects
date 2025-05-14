import cv2
import numpy as np
from PIL import Image

def getColorLimits(color):
    """Helper function to get color limits for masking"""
    colorArray = np.uint8([[color]])  # Insert color to be converted to HSV
    hsvColor = cv2.cvtColor(colorArray, cv2.COLOR_BGR2HSV)

    # For red, we need a wider range since it spans the hue spectrum
    lowerLimit = hsvColor[0][0][0] - 10, 100, 100
    upperLimit = hsvColor[0][0][0] + 10, 255, 255

    # Handle hue wrap-around for red (which is at the edge of the hue spectrum)
    if lowerLimit[0] < 0:
        # Create two ranges for red (since it wraps around 0)
        lowerLimit1 = np.uint8([[[0, lowerLimit[1], lowerLimit[2]]]])
        upperLimit1 = np.uint8([[[upperLimit[0] % 180, upperLimit[1], upperLimit[2]]]])
        lowerLimit2 = np.uint8([[[180 + lowerLimit[0], lowerLimit[1], lowerLimit[2]]]])
        upperLimit2 = np.uint8([[[180, upperLimit[1], upperLimit[2]]]])
        return lowerLimit1, upperLimit1, lowerLimit2, upperLimit2
    else:
        lowerLimit = np.uint8([[[lowerLimit[0] % 180, lowerLimit[1], lowerLimit[2]]]])
        upperLimit = np.uint8([[[upperLimit[0] % 180, upperLimit[1], upperLimit[2]]]])
        return lowerLimit, upperLimit

def main():
    # Define the red color in BGR format
    redColor = [0, 0, 255]  # red in BGR colorspace

    try:
        # Initialize video capture (0 for default camera)
        videoCapture = cv2.VideoCapture(0)

        if not videoCapture.isOpened():
            print("Error: Could not open video capture")
            return

        while True:
            ret, frame = videoCapture.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Flip the frame horizontally (mirror effect)
            frame = cv2.flip(frame, 1)  # 1 = horizontal flip

            # Convert frame to HSV color space
            hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Get color limits for red
            colorLimits = getColorLimits(redColor)

            if len(colorLimits) == 4:
                # For red, we need to handle two ranges
                lowerLimit1, upperLimit1, lowerLimit2, upperLimit2 = colorLimits
                mask1 = cv2.inRange(hsvImage, lowerLimit1, upperLimit1)
                mask2 = cv2.inRange(hsvImage, lowerLimit2, upperLimit2)
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                lowerLimit, upperLimit = colorLimits
                mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

            # Convert mask to PIL Image
            maskImage = Image.fromarray(mask)

            # Get bounding box
            boundingBox = maskImage.getbbox()

            if boundingBox is not None:
                x1, y1, x2, y2 = boundingBox
                # Draw rectangle around detected red object
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)  # Red rectangle

            # Show the frame with detection
            cv2.imshow('Red Color Detection', frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as error:
        print(f"An error occurred: {error}")

    finally:
        # Release resources
        if 'videoCapture' in locals():
            videoCapture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
