import sys
import cv2

RESIM_YOLU = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

resim1 = cv2.imread(RESIM_YOLU, cv2.IMREAD_GRAYSCALE)
resim1 = cv2.resize(resim1, (0, 0), fx=0.5, fy=0.5)

# ikinci resim birincinin ortasindan kirpiliyor
h, w = resim1.shape
resim2 = resim1[int(h*0.1):int(h*0.8), int(w*0.1):int(w*0.8)]

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(resim1, None)
kp2, des2 = sift.detectAndCompute(resim2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
eslesmeler = bf.match(des1, des2)

eslesmeler = sorted(eslesmeler, key=lambda x: x.distance)

# en iyi 25 eslesme
resim_eslesme = cv2.drawMatches(
    resim1, kp1,
    resim2, kp2,
    eslesmeler[:25], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imshow("SIFT ile Ozellik Eslestirme", resim_eslesme)

cv2.waitKey(0)
cv2.destroyAllWindows()