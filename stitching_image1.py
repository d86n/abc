import cv2
import numpy as np

class Stitcher:
    def __init__(self):
        # dùng SIFT để phát hiện đặc trưng
        self.sift = cv2.SIFT_create()

    def stitch(self, images, showMatches=False):
        (imageB, imageA) = images

        # lấy keypoints và descriptors
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # tìm homography
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB)
        if M is None:
            return None, None

        (matches, H, status) = M

        # warp ảnh A theo ảnh B
        (hB, wB) = imageB.shape[:2]
        (hA, wA) = imageA.shape[:2]

        result = cv2.warpPerspective(imageA, H, (wB, hB))

        # trộn ảnh (không mở rộng canvas)
        mask = (result == 0)
        result[mask] = imageB[mask]

        vis = None
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)

        return result, vis

    def detectAndDescribe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kps, features = self.sift.detectAndCompute(gray, None)
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio=0.75, reprojThresh=4.0):
        matcher = cv2.BFMatcher()
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        for m, n in rawMatches:
            if m.distance < n.distance * ratio:
                matches.append(m)

        if len(matches) > 4:
            ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
            ptsB = np.float32([kpsB[m.trainIdx] for m in matches])

            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            return (matches, H, status)

        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        for ((match), s) in zip(matches, status):
            if s == 1:
                ptA = (int(kpsA[match.queryIdx][0]), int(kpsA[match.queryIdx][1]))
                ptB = (int(kpsB[match.trainIdx][0]) + wA, int(kpsB[match.trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        return vis


if __name__ == "__main__":
    img1 = cv2.imread("C:\\Users\\ADMIN\\OneDrive\\Documents\\abc\\abc\\photos\\query.jpg")  # ảnh từ camera 1
    img2 = cv2.imread("C:\\Users\\ADMIN\\OneDrive\\Documents\\abc\\abc\\photos\\train.jpg")  # ảnh từ camera 2

    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([img1, img2], showMatches=True)

    if result is None:
        print("Không tìm được Homography!")
    else:
        cv2.imwrite("panorama.png", result)
        if vis is not None:
            cv2.imwrite("matches.png", vis)
