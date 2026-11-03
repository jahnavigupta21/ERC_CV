
import cv2
import mediapipe as mp
import time
import pygame

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

#counting the number of fingers raised
def count_fingers(hand_landmarks,label):

    count = 0

    #Check the thumb
    thumb_tip = hand_landmarks[4].x
    thumb_joint = hand_landmarks[3].x

    if label=="Right":
        if(thumb_tip < thumb_joint):
            count+=1

    elif label=="Left":
        if(thumb_tip>thumb_joint):
            count+=1      

    # Fingertip and PIP joint indices
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    # Check whether each of the four fingers is raised
    for tip, pip in zip(finger_tips, finger_pips):

        if hand_landmarks[tip].y < hand_landmarks[pip].y:
            count += 1

    return count




# open the webcam
cap = cv2.VideoCapture(0)

# record when video processing starts( we need it beacuse mediapipe VIDEO detector expects a timestamp for every frame)
start_time = time.time()

#define hand_connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]



pygame.mixer.init()

sounds_mapping = {
    1: pygame.mixer.Sound("sounds_swaras/1_Sa.mp3"),
    2: pygame.mixer.Sound("sounds_swaras/2_Re.mp3"),
    3: pygame.mixer.Sound("sounds_swaras/3_Ga.mp3"),
    4: pygame.mixer.Sound("sounds_swaras/4_Ma.mp3"),
    5: pygame.mixer.Sound("sounds_swaras/5_Pa.mp3"),
    6: pygame.mixer.Sound("sounds_swaras/6_Dha.mp3"),
    7: pygame.mixer.Sound("sounds_swaras/7_Ni.mp3"),
    8: pygame.mixer.Sound("sounds_swaras/8_Sa_upper.mp3")
}

#Todo 2 : Detect hands in every frame

previous_count=-1
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

# Count raised fingers
    total_fingers = 0

    for i,hand_landmarks in enumerate(result.hand_landmarks):

        # Get whether this hand is Left or Right
        label = result.handedness[i][0].category_name


        fingers = count_fingers(hand_landmarks,label)

        total_fingers += fingers

    if total_fingers != previous_count:

        if total_fingers in sounds_mapping:
            sounds_mapping[total_fingers].play()

        previous_count = total_fingers

# Display the total number of raised fingers
    print("Fingers raised:", total_fingers)

    cv2.putText(
    frame,
    f"Fingers: {total_fingers}",
    (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 0, 0),
    2
    )

#display the webcam
    cv2.imshow("Hand Gesture Music Player", frame)

#exit when q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


#release resources
cap.release()
cv2.destroyAllWindows()
landmarker.close()
pygame.mixer.quit()






