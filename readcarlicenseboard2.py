import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

# readcarlicenseboard0.py
# numberboard.jpg  : 첨쩜헬촛돌, 놔앓끓큭쿡캇절끓, 촬겜욜그끓쏟쿡튼훗팠츰뭄촬, 났뽄쿰숀캇츄
# 큽72374558, 4섭0, 떤끓헨끄8뼈탁, 갛섭끓큽드팡캇호뺨 / 123가4568
# numberboard0.jpg : 츰65노0887, 삐 / 65노0887
# numberboard1.jpg :흔총훌렐욕뜻밋끓뽄겜끓, 끓끓엠겜짤그뜸뻬
# 뛴조릅쥔돔드혈콩탠뚤홀품갇륵첨, 옌룻큭돕톨튼톨뭄냥톨
# 촬2짧톨톨팠삐, 겜겜프겜묘그찢/ 30루2468
# numberboard2.jpg : 뭄겜겜, 흉헬21큽남234롤쩨룰 / 21BH2345AA
# numberboard3.jpg : 19오7777 / 19오7777

# 이미지 불러오기
img = cv2.imread('numberboard3.jpg')
height, width, channel = img.shape

# 회색으로
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 모폴로지연산을 위한 사각 커널 생성
# 모폴로지 연산이란?
# 주로 이진 이미지의 형태나 구조를 분석하고 처리하는 연산
# 이미지의 특정 부분을 확대하거나 축소하는데 사용된다.

# 침식(Erosion)
#   이미지의 경계를 축소하거나 얇게 만듬
#   구조 요소가 이미지의 픽셀 위를 지나갈 때, 
#   구조 요소에 완전히 포함되는 픽셀만 남기고 나머지는 제거
#   이는 작은 물체를 제거하거나 물체 간의 간격을 늘리는 데 유용

# 팽창(Dilation)
#   침식의 반대로, 이미지의 경계를 확장
#   구조 요소가 이미지의 픽셀 위를 지나갈 때
#   구조 요소에 닿는 모든 픽셀을 확장
#   이는 구멍을 메우거나 물체를 더 크게 만드는 데 사용

# 열기(Opening)
#   먼저 침식을 수행한 후 팽창을 수행
#   이는 작은 객체나 돌기를 제거하는 데 유용
#   물체의 형태를 보존하면서 노이즈를 제거

# 닫기(Closing)
#   먼저 팽창을 수행한 후 침식을 수행
#   이는 작은 구멍이나 균열을 메우는 데 유용
structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# tophat
# 원본 이미지에서 그 이미지를 열기 연산으로 처리한 결과를 뺀 것
# 원본이미지 - (원본이미지(침식 -> 팽창))
# 원본 이미지에 있었지만 열기 연산에서 제거된 작은 객체나 돌기를 강조

# 노이즈 제거: 이미지에서 작은 밝은 물체나 노이즈를 제거할 때 유용
# 세부 강조: 원본 이미지에 있는 작은 밝은 세부 사항이나 객체를 강조할 때 사용
imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)

# blackhat
# 원본 이미지에서 그 이미지를 닫기 연산으로 처리한 결과를 뺀 것
# 원본이미지 - (원본이미지(팽창 -> 침식))
# 어두운 세부 정보 강조: 이미지에서 작은 어두운 물체나 세부 사항을 강조할 때 유용
# 밝은 배경에서의 어두운 객체 강조: 밝은 배경에 있는 작은 어두운 객체를 더 잘 드러내고 싶을 때 사용
imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

# 그레이스케일과 톱햇 이미지를 합쳐서
# 밝은 부분을 강조.
# 그 결과 작은 세부사항을 강조함.
imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
# 그레이스케일과 톱햇 이미지를 합친 것에서
# 블랙햇 이미지를 뺌
# 어두운 부분을 약화시킴.
# 밝은 부분이 더욱 강조됨.
gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

# 잡티제거 (흐리게)
bulered = cv2.GaussianBlur(gray, ksize=(5,5), sigmaX=0)

# 샤픈을 위한 커스텀 커널
kernel = np.array([[0, -1, 0],
                  [-1, 5, -1],
                  [0, -1, 0]])
# 샤픈으로 테두리 강화
sharpened = cv2.filter2D(bulered, -1, kernel=kernel)

# 쓰레쉬홀드 2진화
threshed = cv2.adaptiveThreshold(sharpened, maxValue=255.0, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=19, C=9)

# 윤곽선 찾기
contours, _ = cv2.findContours(threshed, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

# 임시값 저장할 넘파이 어레이 (커널형)
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

# 외곽선 그리고 한번 찍어보자.
cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color= (255, 255, 255))
cv2.imshow('tempresult', temp_result)
cv2.waitKey(100000)
cv2.destroyAllWindows()

# 변수 초기화
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

# 사각형 범위 찾을 공간 확보
contours_dict = []

# 찾기 시작
for contour in contours:
    # 각 요소의 꼭지점과 상하좌우 사이즈 획득
    x, y, w, h = cv2.boundingRect(contour)
    # 사각형 생성 : 그릴 곳, 좌상단 꼭지점, 우하단 꼭지점 선 색, 선 굵기
    cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255,255,255), thickness=2)

    # dict에 넣기
    contours_dict.append({'contor':contour, 'x':x, 'y':y, 'w':w, 'h':h, 'cx':x+(w/2), 'cy':y+(h/2)})

#어떤 것이 번호판 처럼 생겼는가
MIN_AREA = 80 # 최소 넓이가 80이상
MIN_WIDTH, MIN_HEIGHT = 2, 8 # 최소 가로2, 최소 세로8
MIN_RATIO, MAX_RATIO = 0.25, 1.0 # 최소비율 0.25, 최대비율 1.0

# 위 조건에 맞는 컨투어 들을 새 공간에 넣는다.
possible_contours = []

cnt = 0
for d in contours_dict:
    area = d['w'] * d['h']
    ratio = d['w'] / d['h']

    if area > MIN_AREA \
    and d['w'] > MIN_WIDTH \
    and d['h'] > MIN_HEIGHT \
    and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        cnt += 1
        possible_contours.append(d)

# 임시공간 다시 초기화
temp_result = np.zeros((height, width, channel), dtype= np.uint8)

# 새로만든 목록 (조건에 맞는 녀석들)에 대해 사각형
for d in possible_contours:
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

# 최대 대각선 길이 배수, 영역 차이, 폭 차이, 높이 차이, 최소 일치 개수를 정의하는 상수들
MAX_DIAG_MULTIPLYER = 5 # 대각선 길이의 최대 배수
MAX_ANGLE_DIFF = 12.0 # 최대 각도 차이
MAX_AREA_DIFF = 0.5 # 최대 영역 차이
MAX_WIDTH_DIFF = 0.8 # 최대 너비 차이
MAX_HEIGHT_DIFF = 0.2 # 최대 높이 차이
MIN_N_MATCHED = 3 # 최소 일치해야 하는 개수

# 주어진 윤곽선 목록에서 문자를 찾는 함수 정의
def find_chars(contour_list):
    matched_result_idx = [] # 일치하는 결과의 인덱스를 저장할 리스트
    
    # 주어진 윤곽선 목록을 순회하며 비교
    for d1 in contour_list:
        matched_contours_idx = [] # 일치하는 윤곽선의 인덱스를 저장할 리스트
        for d2 in contour_list:
            if d1['idx'] == d2['idx']: # 동일한 윤곽선을 비교하지 않음
                continue

            # 두 윤곽선 간의 x, y 좌표 차이 계산
            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            # 첫 번째 윤곽선의 대각선 길이 계산
            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            # 두 윤곽선 중심간의 거리 계산
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            # 각도 차이 계산
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            # 영역 차이 계산
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            # 너비 차이 계산
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            # 높이 차이 계산
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            # 윤곽선이 일치하는 조건을 검사
            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        # 현재 윤곽선 추가
        matched_contours_idx.append(d1['idx'])

        # 최소 일치 개수보다 작으면 계속
        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        # 결과에 일치하는 윤곽선 인덱스 추가
        matched_result_idx.append(matched_contours_idx)

        # 일치하지 않는 윤곽선 인덱스를 찾음
        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        # 일치하지 않는 윤곽선을 추출
        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
        
        # 재귀적으로 다시 찾기
        recursive_contour_list = find_chars(unmatched_contour)
        
        # 재귀 결과를 결과 목록에 추가
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break # 첫 번째 일치 그룹을 찾으면 반복 종료

    return matched_result_idx # 일치하는 결과 인덱스 반환
    
# 가능한 윤곽선으로부터 결과 인덱스 찾기
result_idx = find_chars(possible_contours)

matched_result = [] # 일치하는 결과를 저장할 리스트
# 결과 인덱스를 이용하여 일치하는 윤곽선 그룹 추출
for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))

# 가능한 윤곽선 시각화
temp_result = np.zeros((height, width, channel), dtype=np.uint8) # 이미지 초기화

# 일치하는 결과 그룹을 이미지에 그리기
for r in matched_result:
    for d in r:
        # 윤곽선 그리기 주석 처리됨
        # cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        # 윤곽선을 사각형으로 그리기
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
# 번호판의 너비와 높이 패딩, 그리고 최소/최대 비율을 설정하는 상수
PLATE_WIDTH_PADDING = 1.3 # 너비 패딩
PLATE_HEIGHT_PADDING = 1.5 # 높이 패딩
MIN_PLATE_RATIO = 3 # 최소 번호판 비율
MAX_PLATE_RATIO = 10 # 최대 번호판 비율

plate_imgs = [] # 크롭된 번호판 이미지를 저장할 리스트
plate_infos = [] # 번호판 정보를 저장할 리스트

# 일치하는 결과에 대해 반복
for i, matched_chars in enumerate(matched_result):
    sorted_chars = sorted(matched_chars, key=lambda x: x['cx']) # 중심 x 좌표에 따라 정렬

    # 번호판 중심 좌표 계산
    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
    
    # 번호판 너비 계산
    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
    
    sum_height = 0 # 높이의 합을 계산하기 위한 변수
    for d in sorted_chars:
        sum_height += d['h'] # 각 문자의 높이를 더함

    # 번호판 높이 계산
    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
    
    # 두 문자 사이의 세로 거리와 대각선 거리 계산
    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
    triangle_hypotenus = np.linalg.norm(
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    )
    
    # 번호판의 기울기 각도 계산
    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
    
    # 이미지 회전을 위한 변환 행렬 계산
    rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
    
    # 원본 이미지를 회전
    img_rotated = cv2.warpAffine(threshed, M=rotation_matrix, dsize=(width, height))
    
    # 회전된 이미지에서 번호판 영역 크롭
    img_cropped = cv2.getRectSubPix(
        img_rotated, 
        patchSize=(int(plate_width), int(plate_height)), 
        center=(int(plate_cx), int(plate_cy))
    )
    
    # 크롭된 이미지의 비율이 지정된 범위 내에 있는지 확인
    if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
        continue
    
    # 조건을 만족하는 경우, 크롭된 이미지와 번호판 정보를 리스트에 추가
    plate_imgs.append(img_cropped)
    plate_infos.append({
        'x': int(plate_cx - plate_width / 2),
        'y': int(plate_cy - plate_height / 2),
        'w': int(plate_width),
        'h': int(plate_height)
    })
# 최장 문자열 인덱스 및 길이 초기화
longest_idx, longest_text = -1, 0
plate_chars = [] # 인식된 번호판 문자를 저장할 리스트

# 크롭된 번호판 이미지에 대해 반복
for i, plate_img in enumerate(plate_imgs):
    # 이미지 크기 조정 및 이진화 처리
    plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
    _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    
    # 번호판의 최소 및 최대 x, y 좌표 초기화
    plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
    plate_max_x, plate_max_y = 0, 0

    # 윤곽선에 대해 반복
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # 영역 및 비율 계산
        area = w * h
        ratio = w / h

        # 조건에 맞는 경우 최소 및 최대 좌표 업데이트
        if area > MIN_AREA \
        and w > MIN_WIDTH and h > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
            if x < plate_min_x:
                plate_min_x = x
            if y < plate_min_y:
                plate_min_y = y
            if x + w > plate_max_x:
                plate_max_x = x + w
            if y + h > plate_max_y:
                plate_max_y = y + h
                
    # 크롭된 이미지 추출
    img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
    
    # 가우시안 블러 및 이진화 처리
    img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
    _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 가장자리에 여백 추가
    img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

    # Tesseract OCR을 사용하여 문자 인식
    chars = pytesseract.image_to_string(img_result, lang='kor+eng', config='--psm 7 --oem 0')
    
    # 필터링된 문자열 저장 및 숫자 포함 여부 확인
    result_chars = ''
    has_digit = False
    for c in chars:
        if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
            if c.isdigit():
                has_digit = True
            result_chars += c
    
    print(result_chars) # 인식된 문자 출력
    plate_chars.append(result_chars) # 인식된 문자를 리스트에 추가

    # 숫자를 포함하고 가장 긴 문자열이면 인덱스 업데이트
    if has_digit and len(result_chars) > longest_text:
        longest_idx = i

    # matplotlib를 사용하여 인식 결과 시각화
    plt.subplot(len(plate_imgs), 1, i+1)
    plt.imshow(img_result, cmap='gray')
    plt.show()
