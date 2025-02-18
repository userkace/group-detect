import cv2
import os

def crop_head_from_video(video_path, output_folder):
    # Load the face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no more frames are available

        # Use different orientation for cases where output is upside down
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Crop the frame to the detected face (head)
            head_crop = frame[y:y+h, x:x+w]

            # Save the cropped image
            output_path = os.path.join(output_folder, f"image{frame_count + 1}.jpg")
            cv2.imwrite(output_path, head_crop)

            frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Processed {frame_count} frames. Images saved in '{output_folder}'.")

# Example usage
video_name = "padilla_rest"  # Name of file w/o extension
video_path = os.path.join(video_name + ".mp4")  # Join .mp4 for file path
output_folder = video_name  # Output folder same as name

crop_head_from_video(video_path, output_folder)