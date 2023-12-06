import cv2
import numpy as np
from PIL import Image
import pytesseract

def detect_and_save_license_plate(image_path):
    # Load the image
    image = cv2.imread(image_path)

    cv2.imshow("image", image)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

    image = cv2.resize(image, None, fx = 0.5, fy = 0.5)

    # Convert to grayscale and apply binary thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

    # Find contours on the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours based on aspect ratio
    contours = [contour for contour in contours if 1.5 < cv2.boundingRect(contour)[2] / cv2.boundingRect(contour)[3] < 4.465]

    # Sort candidate contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Assuming the largest contour with the correct aspect ratio to be the license plate
    if contours:
        plate_contour = contours[0]
        x, y, w, h = cv2.boundingRect(plate_contour)
        roi = binary[y:y+h, x:x+w]

        # Save the detected license plate region as a separate image
        license_plate_image_path = 'detected_license_plate.jpg'
        cv2.imwrite(license_plate_image_path, roi)
        temp = cv2.imread(license_plate_image_path)
        cv2.imshow("image", temp)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()

        return license_plate_image_path
    else:
        return None
    
def ocr_on_saved_plate(image_path):
    # Load the saved license plate image
    roi = cv2.imread(image_path)

    # Perform OCR on the image
    pil_roi = Image.fromarray(roi)
    return pytesseract.image_to_string(pil_roi, lang='kor+eng',config='--psm 8')

# Example Usage
image_path = 'numberboard3.jpg'
roi = detect_and_save_license_plate(image_path)
if roi is not None:
    text = ocr_on_saved_plate(roi)
    print("Detected License Plate Text:", text)
else:
    print("License plate not found.")


# readcarlicenseboard0.py
# numberboard.jpg  : 1237 4560 / 123가4568
# numberboard0.jpg : _{_:줏숟낮낮′돔 / 65노0887
# numberboard1.jpg :  으 / 30루2468
# numberboard2.jpg : 、 21 BH 2345 42 / 21BH2345AA
# numberboard3.jpg : 놀5 / 19오7777