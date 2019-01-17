import numpy as np
import cv2 as cv


width_height = 0
'''
사진의 위치. 
30 31 32 33 34
20 21 22 23 24
10 11 12 13 14
00 01 02 03 04

이런 큰 틀 안에서 사진의 이름을 그때그때 정해주었다. (00 기준)
단, 1 * 10일 때는 00~14로 해주었다.
'''

# Image read
f00 = cv.imread('00.jpeg', 1)
f01 = cv.imread('01.jpeg', 1)
f02 = cv.imread('02.jpeg', 1)
f03 = cv.imread('03.jpeg', 1)
f04 = cv.imread('04.jpeg', 1)
f10 = cv.imread('10.jpeg', 1)
f11 = cv.imread('11.jpeg', 1)
f12 = cv.imread('12.jpeg', 1)
f13 = cv.imread('13.jpeg', 1)
f14 = cv.imread('14.jpeg', 1)
f20 = cv.imread('20.jpeg', 1)
f21 = cv.imread('21.jpeg', 1)
f22 = cv.imread('22.jpeg', 1)
f23 = cv.imread('23.jpeg', 1)
f24 = cv.imread('24.jpeg', 1)
f30 = cv.imread('30.jpeg', 1)
f31 = cv.imread('31.jpeg', 1)
f32 = cv.imread('32.jpeg', 1)
f33 = cv.imread('33.jpeg', 1)
f34 = cv.imread('34.jpeg', 1)


def panorama(f1, f2): # matching점을 찾아서 사진들을 stitching 해주는 함수 ( 주 함수 )
    # initiate sift detector

    #orb 객체를 생성
    orb = cv.ORB_create()

    # 이미지 1,2의 키포인트와 디스크립터 계산
    kp1, des1 = orb.detectAndCompute(f1, None)
    kp2, des2 = orb.detectAndCompute(f2, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2) # 디스크립터 matching, 결과 matches에 담음

     # 매칭점들을 이어주기 위한 코드. 시각화할 수 있으나, 프로젝트를 구현하는데에 있어서는 필요함을 느끼지 못하고 사용하지 않았다.
    matches = sorted(matches, key = lambda x:x.distance)
    res = None
    res = cv.drawMatches(f1, kp1, f2, kp2, matches[:10], outImg=res, flags=2)

    # keypoint를 numpy로 바꾸었다.
    kp1 = np.float32([kp.pt for kp in kp1])
    kp2 = np.float32([kp.pt for kp in kp2])

    # raw Matches
    matcher = cv.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(des1, des2, 2)
    matchh = []

    # loop over the raw matches
    for m in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            matchh.append((m[0].trainIdx, m[0].queryIdx))

    if len(matchh) > 4:
        # xy_1, xy_2에 key point를 numnpy로 할당해준다.
        xy_1 = np.float32([kp1[i] for (_, i) in matchh])
        xy_2 = np.float32([kp2[i] for (i, _) in matchh])

   # print(len(xy_1), len(xy_2))
    # Get homography matrix 찾는함수
    # H = cv.getPerspectiveTransform(xy_1.astype(np.float32), xy_2.astype(np.float32))

    # 전에 사용한 getPerspectivTransform과 달리, RANSAC를 사용하기 위하여
    # findHomography라를 함수를 사용하였다.
    (H, status) = cv.findHomography(xy_1, xy_2, cv.RANSAC, 4.0)

    # inverse
    Hinv = np.linalg.inv(H)

    # 사진들을 가로로 붙일 때에는 width_height를 0으로, 세로로 붙일 때에는 1로 놓고,
    # 각각 result가 되는 사진들의 총 크기를 0일때에는 가로를 본 사진의 2배로, 1일 때에는 세로를 본 사진의 2배로 되도록 하였다.
    if (width_height == 0):
        result = cv.warpPerspective(f2, Hinv, (f2.shape[1] + f1.shape[1], f2.shape[0]))
    if (width_height == 1):
        result = cv.warpPerspective(f2, Hinv, (f2.shape[1], f2.shape[0] + f2.shape[0]))

    #최종 결과물을 붙이고
    result[0:f1.shape[0], 0:f1.shape[1]] = f1

    # 결과 result와, 매칭점을 리턴한다.
    # 매칭점같은경우 매칭되는 점들을 보여주고 싶으면 사용 가능하나,
    # 꼭 보여주지 않아도 된다고 판단하여 생략하였다.


    return (result, xy_1, xy_2)


# 검정색 부분과 겹쳐지지 않도록 생각하며
# 한칸한칸 사진을 붙였다.
# 이미지의 tone 수정 (gamma값으로)
def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv.LUT(image, table)


# 각각의 모든 이미지를 gray image로
gray_image_f00 = cv.cvtColor(f00, cv.COLOR_BGR2GRAY)
gray_image_f01 = cv.cvtColor(f01, cv.COLOR_BGR2GRAY)
gray_image_f02 = cv.cvtColor(f02, cv.COLOR_BGR2GRAY)
gray_image_f10 = cv.cvtColor(f10, cv.COLOR_BGR2GRAY)
gray_image_f11 = cv.cvtColor(f11, cv.COLOR_BGR2GRAY)
gray_image_f12 = cv.cvtColor(f12, cv.COLOR_BGR2GRAY)
gray_image_f20 = cv.cvtColor(f20, cv.COLOR_BGR2GRAY)
gray_image_f21 = cv.cvtColor(f21, cv.COLOR_BGR2GRAY)
gray_image_f22 = cv.cvtColor(f22, cv.COLOR_BGR2GRAY)
gray_image_f30 = cv.cvtColor(f30, cv.COLOR_BGR2GRAY)
gray_image_f31 = cv.cvtColor(f31, cv.COLOR_BGR2GRAY)
gray_image_f32 = cv.cvtColor(f32, cv.COLOR_BGR2GRAY)


# 각각의 gray image의 평균
gray_mean_f00 = np.mean(gray_image_f00)
gray_mean_f01 = np.mean(gray_image_f01)
gray_mean_f02 = np.mean(gray_image_f02)
gray_mean_f10 = np.mean(gray_image_f10)
gray_mean_f11 = np.mean(gray_image_f11)
gray_mean_f12 = np.mean(gray_image_f12)
gray_mean_f20 = np.mean(gray_image_f20)
gray_mean_f21 = np.mean(gray_image_f21)
gray_mean_f22 = np.mean(gray_image_f22)
gray_mean_f30 = np.mean(gray_image_f30)
gray_mean_f31 = np.mean(gray_image_f31)
gray_mean_f32 = np.mean(gray_image_f32) # 기준


# 맨 처음 이미지(f32)와 다른 모든 이미지들의 차이
# 차이가 gamma 값이 된다. 즉, 차이만큼 tone을 수정한다.
gamma_f32_31 = gray_mean_f32 / gray_mean_f31
gamma_f32_30 = gray_mean_f32 / gray_mean_f30
gamma_f32_22 = gray_mean_f32 / gray_mean_f22
gamma_f32_21 = gray_mean_f32 / gray_mean_f21
gamma_f32_20 = gray_mean_f32 / gray_mean_f20
gamma_f32_12 = gray_mean_f32 / gray_mean_f12
gamma_f32_11 = gray_mean_f32 / gray_mean_f11
gamma_f32_10 = gray_mean_f32 / gray_mean_f10
gamma_f32_02 = gray_mean_f32 / gray_mean_f02
gamma_f32_01 = gray_mean_f32 / gray_mean_f01
gamma_f32_00 = gray_mean_f32 / gray_mean_f00


'''
gamma_f32_31 = gray_mean_f31 / gray_mean_f32
gamma_f32_30 = gray_mean_f30 / gray_mean_f32
gamma_f32_22 = gray_mean_f22 / gray_mean_f32
gamma_f32_21 = gray_mean_f21 / gray_mean_f32
gamma_f32_20 = gray_mean_f20 / gray_mean_f32
gamma_f32_12 = gray_mean_f12 / gray_mean_f32
gamma_f32_11 = gray_mean_f11 / gray_mean_f32
gamma_f32_10 = gray_mean_f10 / gray_mean_f32
gamma_f32_02 = gray_mean_f02 / gray_mean_f32
gamma_f32_01 = gray_mean_f01 / gray_mean_f32
gamma_f32_00 = gray_mean_f00 / gray_mean_f32
'''
# 위에서 구한 차이로 tone을 adjust한다.
adjusted_f31 = adjust_gamma(f31, gamma=gamma_f32_31)
adjusted_f30 = adjust_gamma(f30, gamma=gamma_f32_30)
adjusted_f22 = adjust_gamma(f22, gamma=gamma_f32_22)
adjusted_f21 = adjust_gamma(f21, gamma=gamma_f32_21)
adjusted_f20 = adjust_gamma(f20, gamma=gamma_f32_20)
adjusted_f12 = adjust_gamma(f12, gamma=gamma_f32_12)
adjusted_f11 = adjust_gamma(f11, gamma=gamma_f32_11)
adjusted_f10 = adjust_gamma(f10, gamma=gamma_f32_10)
adjusted_f02 = adjust_gamma(f02, gamma=gamma_f32_02)
adjusted_f01 = adjust_gamma(f01, gamma=gamma_f32_01)
adjusted_f00 = adjust_gamma(f00, gamma=gamma_f32_00)

print(gamma_f32_31)
print(gamma_f32_30)
print(gamma_f32_22)
print(gamma_f32_21)
print(gamma_f32_20)
print(gamma_f32_12)
print(gamma_f32_11)
print(gamma_f32_10)
print(gamma_f32_02)
print(gamma_f32_01)
print(gamma_f32_00)

(result,xy_1,xy_2)=panorama(adjusted_f31,f32)
(result,xy_1,xy_2)=panorama(adjusted_f30,result)


(result1,xy_1,xy_2)=panorama(adjusted_f21,adjusted_f22)
(result1,xy_1,xy_2)=panorama(adjusted_f20,result1)


(result2,xy_1,xy_2)=panorama(adjusted_f11,adjusted_f12)
(result2,xy_1,xy_2)=panorama(adjusted_f10,result2)


(result3,xy_1,xy_2)=panorama(adjusted_f01,adjusted_f02)
(result3,xy_1,xy_2)=panorama(adjusted_f00,result3)

width_height=1
(result0,xy_1,xy_2)=panorama(result2,result3)
(result0,xy_1,xy_2)=panorama(result1,result0)
(result0,xy_1,xy_2)=panorama(result,result0)
'''
#4*3
width_height=0

(result,xy_1,xy_2)=panorama(f31,f32)
(result,xy_1,xy_2)=panorama(f30,result)


(result1,xy_1,xy_2)=panorama(f21,f22)
(result1,xy_1,xy_2)=panorama(f20,result1)


(result2,xy_1,xy_2)=panorama(f11,f12)
(result2,xy_1,xy_2)=panorama(f10,result2)


(result3,xy_1,xy_2)=panorama(f01,f02)
(result3,xy_1,xy_2)=panorama(f00,result3)

width_height=1
(result0,xy_1,xy_2)=panorama(result2,result3)
(result0,xy_1,xy_2)=panorama(result1,result0)
(result0,xy_1,xy_2)=panorama(result,result0)
'''
cv.imshow('result', result0);

cv.imwrite('./results/result.jpg', result0)
cv.waitKey(0)



'''
1*10
(result, xy_1, xy_2) = panorama(f13, f14)
(result, xy_1, xy_2) = panorama(f12, result)
(result, xy_1, xy_2) = panorama(f11, result)
(result, xy_1, xy_2) = panorama(f10, result)
(result, xy_1, xy_2) = panorama(f04, result)
(result, xy_1, xy_2) = panorama(f03, result)
(result, xy_1, xy_2) = panorama(f02, result)
(result, xy_1, xy_2) = panorama(f01, result)
(result, xy_1, xy_2) = panorama(f00, result)
'''

''' 
#3*4
width_height=0
(result,xy_1,xy_2)=panorama(f22,f23)
(result,xy_1,xy_2)=panorama(f21,result)
(result,xy_1,xy_2)=panorama(f20,result)

(result1,xy_1,xy_2)=panorama(f12,f13)
(result1,xy_1,xy_2)=panorama(f11,result1)
(result1,xy_1,xy_2)=panorama(f10,result1)

(result2,xy_1,xy_2)=panorama(f02,f03)
(result2,xy_1,xy_2)=panorama(f01,result2)
(result2,xy_1,xy_2)=panorama(f00,result2)

width_height=1
(result0,xy_1,xy_2)=panorama(result1,result2)
(result0,xy_1,xy_2)=panorama(result,result0)

'''
