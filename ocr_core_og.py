
import cv2
import imutils
import numpy as np
import pytesseract


def ocr_core_og(filename):
    # Reads the image and resizes it
    img = cv2.imread(filename,cv2.IMREAD_COLOR)
    img = cv2.resize(img, (600,400) )

    # Grayscales
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray = cv2.bilateralFilter(gray, 13, 55, 55)

    # Binarizes the image 
    edged = cv2.Canny(gray, 30, 200)

    # Grabs the counters using a rectangular box and crops that
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None

    # Find all the enclosures, and iterate a for loop to find a rectangular enclosure
    for c in contours:
        
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        detected = 0
        print ("No contour detected")
    else:
        detected = 1

    # Only the number plate is made visible, and rest is masked
    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
        mask = np.zeros(gray.shape,np.uint8)
        new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
        new_image = cv2.bitwise_and(img,img,mask=mask)

    # The masked area is cropped and individual characters are segmented
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx+1, topy:bottomy+1]

    # Pytesseract uses OCR here to recognise these charaters
        text = pytesseract.image_to_string(Cropped, config='--psm 11')
        img = cv2.resize(img,(500,300))
        Cropped = cv2.resize(Cropped,(400,200))
        return text
        # cv2.imshow('car',img)
        # cv2.imshow('Cropped',Cropped)

    else:
        error_text="Please click a clearer photo"
        return error_text

    # Closes the program
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# finalanswer=ocr_core_og('007.jpg')
# print(finalanswer)
# print("hi")