# Import the necessary libraries
import cv2              # Computer Vision
import numpy as np      # Numpy

# An empty function for the Trackbars
def nothing(x):
    pass


def main():
    # Open webcam (0 = default camera)
    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("H_min", "Trackbars", 100, 179, nothing)
    cv2.createTrackbar("H_max", "Trackbars", 140, 179, nothing)
    cv2.createTrackbar("S_min", "Trackbars", 150, 255, nothing)
    cv2.createTrackbar("S_max", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("V_min", "Trackbars", 50, 255, nothing)
    cv2.createTrackbar("V_max", "Trackbars", 255, 255, nothing)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get current positions of trackbars
        h_min = cv2.getTrackbarPos("H_min", "Trackbars")
        h_max = cv2.getTrackbarPos("H_max", "Trackbars")
        s_min = cv2.getTrackbarPos("S_min", "Trackbars")
        s_max = cv2.getTrackbarPos("S_max", "Trackbars")
        v_min = cv2.getTrackbarPos("V_min", "Trackbars")
        v_max = cv2.getTrackbarPos("V_max", "Trackbars")

        # Define color range (BLUE object)
        lower_blue = np.array([h_min, s_min, v_min])
        upper_blue = np.array([h_max, s_max, v_max])

        # Create mask
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Noise removal
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove small objects
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Take largest contour
            largest = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest) > 500:
                x, y, w, h = cv2.boundingRect(largest)
                cx = x + w // 2
                cy = y + h // 2

                # Draw bounding box and centroid
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        cv2.imshow("Color Tracking", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
