
import cv2
import mediapipe as mp
import time


from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision



#Path to the hand detection model
MODEL_PATH = "hand_landmarker.task"

# Todo 1: Create the hand detector

# Tell mediapipe where the model is created
base_options = mp_python.BaseOptions(
    model_asset_path=MODEL_PATH
)

# configure the detector
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2,
)

# create the detector(we are creating the detector outside the loop as loading the model is computationally expensive 
# and loading the model every time in a loop makes the program run slower)
landmarker = mp_vision.HandLandmarker.create_from_options(
    options
)

# open the webcam
cap = cv2.VideoCapture(0)

# record when video processing starts( we need it beacuse mediapipe VIDEO detector expects a timestamp for every frame)
start_time = time.time()

#Todo 2 : Detect hands in every frame


while True:

    #read one frame from the webcam
    success, frame = cap.read()

    if not success:
        print("Could not access the webcam.")
        break
#mirror the webcam image( 1 means flipping horizontally)
    frame = cv2.flip(frame, 1)


# Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Prepare the rgb image into a  MediaPipe image( OpenCV gives an image array so we convert it to mp.image)
    mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_frame
    )

# Calculate the timestamp in milliseconds
    timestamp_ms = int((time.time() - start_time) * 1000)

# Detect hands in this frame
   
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
# draw landmarks for detected hands

if result.hand_landmarks:
    for hand_landmarks in result.hand_landmarks:
        draw_landmarks(frame, hand_landmarks)
#print the number of detected hands
    print(len(result.hand_landmarks))



# Todo 3 : draw the landmarks on the hands
def draw_landmarks(frame, hand_landmarks):
        
#get the height and width of the frame  
    h, w, _ = frame.shape
    
    
    

# Convert each landmark to pixel coordinates
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

# Draw lines between connected landmarks (green)
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (0,255,0), 2)

# Draw a dot at each landmark (red, filled)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0,0,255), -1)

    if hand_landmarks[8].y < hand_landmarks[6].y:
     

#display the webcam
    cv2.imshow("Hand Gesture Music Player", frame)

#exit when q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


#release resources
cap.release()
cv2.destroyAllWindows()
landmarker.close()







