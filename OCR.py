import cv2
import pytesseract
import numpy as np
import pandas as pd



def OCR(video_file):
    # Configure pytesseract path to where the tesseract executable is located
    pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/5.3.4_1/bin/tesseract'

    def extract_text_from_frame(frame):
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # Apply Adaptive Thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # (Optional) Dilation to enhance the edges
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        # Use Tesseract to extract text
        text = pytesseract.image_to_string(gray_frame, lang='eng')
        return text

    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Define the region of interest (x, y, width, height)
    roi_points = (500, 0, 250, 100)
    roi_time = (800, 0, 200, 100)
    roi_accuracy = (1100, 0, 200, 100)

    points = []
    time = []
    accuracy = []

    # Determine the video's frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define how many times per second you want to update
    updates_per_second = 2  # For example, update twice per second

    # Calculate how many frames to skip based on the original video frame rate
    frames_to_skip = int(fps / updates_per_second)

    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % frames_to_skip == 0:
            # Crop to the region of interest
            roi_frame_points = frame[roi_points[1]:roi_points[1]+roi_points[3], roi_points[0]:roi_points[0]+roi_points[2]]
            roi_frame_time = frame[roi_time[1]:roi_time[1] + roi_time[3], roi_time[0]:roi_time[0] + roi_time[2]]
            roi_frame_accuracy = frame[roi_accuracy[1]:roi_accuracy[1] + roi_accuracy[3], roi_accuracy[0]:roi_accuracy[0] + roi_accuracy[2]]

            # Extract text from the cropped region
            extracted_text_points = extract_text_from_frame(roi_frame_points)
            extracted_text_time = extract_text_from_frame(roi_frame_time)
            extracted_text_accuracy = extract_text_from_frame(roi_frame_accuracy)

            points_int = ""
            time_int = ""
            accuracy_int = ""

            for char in extracted_text_points:
                if char.isdigit():
                    points_int += char

            for char in extracted_text_time:
                if char.isdigit():
                    time_int += char

            for char in extracted_text_accuracy:
                if char.isdigit():
                    accuracy_int += char

            points.append(points_int)
            time.append(time_int)
            accuracy.append(accuracy_int)

            print(points_int, time_int, accuracy_int)

        current_frame += 1

    table = {'points': points, 'time': time, 'accuracy': accuracy}

    csv_file = f"{video_file}.csv"
    OCR_data = pd.DataFrame.from_dict(table)
    OCR_data.to_csv(csv_file)

    cap.release()
    cv2.destroyAllWindows()

    return csv_file
