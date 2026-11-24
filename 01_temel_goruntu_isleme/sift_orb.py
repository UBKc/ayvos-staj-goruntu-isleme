import sys
import cv2

RESIM_YOLU = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

resim_gri = cv2.imread(RESIM_YOLU, cv2.IMREAD_GRAYSCALE)
resim_gri = cv2.resize(resim_gri, (0, 0), fx=0.5, fy=0.5)

sift = cv2.SIFT_create()
kp_sift, des_sift = sift.detectAndCompute(resim_gri, None)

# daireler olcek ve yonu gosteriyor
resim_sift = cv2.drawKeypoints(
    resim_gri,
    kp_sift,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

orb = cv2.ORB_create()
kp_orb, des_orb = orb.detectAndCompute(resim_gri, None)

resim_orb = cv2.drawKeypoints(
    resim_gri,
    kp_orb,
    None,
    color=(0, 255, 0),
    flags=0
)

cv2.imshow("1 - SIFT Kilit Noktalari", resim_sift)
cv2.imshow("2 - ORB Kilit Noktalari", resim_orb)

cv2.waitKey(0)
cv2.destroyAllWindows()