import cv2

harcascade = "model/haarcascade_russian_plate_number.xml"
# Set Up Webcam  # Initialize webcam (0 for default webcam, change if needed ex. 1, 2)
cap = cv2.VideoCapture(0) 

cap.set(3, 640) # width
cap.set(4, 480) #height

min_area = 500
count = 0

# Main Loop to Capture Frames and Detect Number Plates
while True:
    # Capture frame-by-frame
    success, img = cap.read()

    plate_cascade = cv2.CascadeClassifier(harcascade)
     # use for convert frame to gray scale format 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=4)
    # scaleFactor - increases the chance of detecting smaller objects and increases processing time
    # minNeighbors - controls neighboring rectangles
    # minSize-detected objects smaller than this size will be ignored

    for (x,y,w,h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
        
        # another frame for detect crop number plates   
            img_roi = img[y: y+h, x:x+w]
            cv2.imshow("ROI", img_roi)

    # Display the frame with detected plates
    cv2.imshow("Result", img)
    
    # Press 's' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # save number plate in plates folder in jpg format
        cv2.imwrite("plates/scaned_img_" + str(count) + ".jpg", img_roi)
        cv2.rectangle(img, (0,200), (640,300), (0,255,0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results",img)
        cv2.waitKey(500)
        count += 1
