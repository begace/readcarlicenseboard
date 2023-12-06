import cv2
import pytesseract
from matplotlib import pyplot as plt
import numpy as np
import re

def rotate_image(image, angle, center):
    # 회전을 위한 변환 행렬
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated

def calculate_angle_and_rotate(contour, img):
    rect = cv2.minAreaRect(contour)
    (x, y), (width, height), angle = rect

    if width < height:
        angle = 90 + angle

    # 회전 중심점 설정
    center = (int(x + width / 2), int(y + height / 2))

    # 이미지 회전
    return rotate_image(img, angle, center), angle

# ROI 이미지들을 하나의 큰 이미지에 배치
def create_collage(rois, rois_per_row=5):
    # ROI의 크기를 동일하게 조정
    resized_rois = [cv2.resize(roi, (100, 50)) for roi in rois]

    # 총 행과 열 계산
    rows = (len(resized_rois) + rois_per_row - 1) // rois_per_row
    cols = min(len(resized_rois), rois_per_row)

    # 빈 캔버스 생성
    collage_height = rows * 50
    collage_width = cols * 100
    collage = np.zeros((collage_height, collage_width), dtype=np.uint8)

    # 각 ROI를 캔버스에 배치
    for idx, roi in enumerate(resized_rois):
        row = idx // rois_per_row
        col = idx % rois_per_row
        collage[row * 50:row * 50 + 50, col * 100:col * 100 + 100] = roi

    return collage

def licenseOCR(img):
    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    # 이진화
    _, thresholded = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY) # cv2.THRESH_OTSU

    # 윤곽선 찾기
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [contour for contour in contours if 1.5 < cv2.boundingRect(contour)[2] / cv2.boundingRect(contour)[3] < 5]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    counter = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # 추출된 이미지마다 기울기 계산 및 회전
        rotated_img, angle = calculate_angle_and_rotate(contour, img)

        # 사각형을 그린 이미지 복사본 생성 (회전된 이미지에 그리기)
        cv2.rectangle(rotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 사각형 옆에 번호 표시 (회전된 이미지에 그리기)
        cv2.putText(rotated_img, str(counter), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        counter += 1

    # 모든 사각형과 번호가 그려진 이미지 표시
    cv2.imshow('Bounding Boxes with Numbers', thresholded)
    cv2.waitKey(100000)  # 100초 동안 대기

    # 모든 윈도우 닫기
    cv2.destroyAllWindows()

    rois = []
    counter = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = thresholded[y:y+h, x:x+w]

        # 추출된 텍스트 출력
        extractedText = pytesseract.image_to_string(roi, config= '--psm 8', lang='kor+eng')
        extractedText = re.sub(r'[^가-힣0-9a-zA-Z]', '', extractedText)
        if len(extractedText) > 6 and bool(re.search(r'\d',extractedText)) : print(f"{counter} : {extractedText}")
        counter += 1

        rois.append(roi)
    # 콜라주 생성 및 표시
    collage = create_collage(rois)
    cv2.imshow('Collage of ROIs', collage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('numberboard3.jpg')
licenseOCR(img)

# readcarlicenseboard1.py
# numberboard.jpg  : 212374569 / 123가4568
# numberboard0.jpg : 65노0887 / 65노0887
# numberboard1.jpg : 솜읊꾼츤4틀욜큭헵 / 30루2468
# numberboard2.jpg : 21BH2345AA, M21BH2345AAIie / 21BH2345AA
# numberboard3.jpg : (못찾음) / 19오7777