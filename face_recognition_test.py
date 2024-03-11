# imports
import face_recognition as fr
import cv2
import numpy as np
import glob

# webcam input number (0 is default builtin webcam)
WEBCAM_INPUT_NUMBER = 0

# number of frames the person needs to be in front of the camera
# for the program to actually make a request
VERIFICATION_FRAMES = 10

# take video capture from source
video_capture = cv2.VideoCapture(WEBCAM_INPUT_NUMBER)

# known face encodings, face names to be stored
known_face_encodings = []
known_face_names = []

# iterate through all pictures in the folder
for filename in glob.glob('pics/*.jpg'):
    print(filename)
    # load image
    im = fr.load_image_file(filename)

    # create encoding of the face
    encoding = fr.face_encodings(im)[0]

    # get person's name be removing prefix, suffix
    person_name = filename.removeprefix('pics\\').removesuffix('.jpg')

    # add encoding, name to list
    known_face_encodings.append(encoding)
    known_face_names.append(person_name)

# initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

prev_name = ''

frame_count = 0

# code taken from 
# https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py

# iterate indefinitely
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    
    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # make it a contiguous array
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Find all the faces and face encodings in the current frame of video
        face_locations = fr.face_locations(rgb_small_frame)
        face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

        # get list for face names
        face_names = []
        # iterate through all encodings
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = fr.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # use the known face with the smallest distance to the new face
            face_distances = fr.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # add to face names
            face_names.append(name)

    # toggle process boolean
    process_this_frame = not process_this_frame


    max_name = ''
    max_area = 0

    found_face = False
    max_top, max_right, max_bottom, max_left = 0, 0, 0, 0

    # iterate through face locations, face names
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # find area of face
        height = abs(top - bottom)
        width = abs(right-left)
        area = height*width

        # if current area > max_area, set the maximum
        # use this to find the closest face out of all the potential faces
        if area > max_area:
            max_area = area
            max_top, max_right, max_bottom, max_left = top, right, bottom, left
            max_name = name
            found_face = True
    
    # if a face has been found, process it
    # otherwise, reset the frame count
    if found_face:
        # set the color of the box being drawn
        color = ()
        
        # if the name is unknown, make it red
        # otherwise, make it proportional to the frame count
        if max_name == 'Unknown':
            color = (0, 0, 255)
        else:
            color = (0, frame_count * (255/VERIFICATION_FRAMES), 250-frame_count*(255/VERIFICATION_FRAMES))

        # draw rectangle to encompass face
        cv2.rectangle(frame, (max_left, max_top), (max_right, max_bottom), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (max_left, max_bottom - 35), (max_right, max_bottom), color, cv2.FILLED)
        # put text
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, max_name, (max_left + 6, max_bottom - 6), font, 1.0, (255, 255, 255), 1)

        # if name is not unknown, process the picture
        # otherwise, continue
        if max_name != 'Unknown':
            # check if the current name is the same as the name of the previous frame
            # if so, process
            # if not, reset hte counter
            if max_name == prev_name:
                # put percentage data on screen
                cv2.putText(frame, (str(min(frame_count*(round(100/VERIFICATION_FRAMES, 2)), 100)) +'%'), (max_left + 6, max_bottom - 60), font, 1.0, (255, 255, 255), 1)
                
                # increment frame count
                frame_count+=1
                # if person is present for the number of verification frames, create order request
                if frame_count == VERIFICATION_FRAMES:
                    print('order from: ', max_name)
            else:
                prev_name = max_name   
                frame_count = 0
        else:
            prev_name = 'Unknown'    
    else:
        frame_count = 0

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()