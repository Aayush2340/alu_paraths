import cv2
import mediapipe as mp
import time

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
        return lmList

    def handOrientation(self, lmList):
        # Check if hand is front or back based on the landmarks
        if lmList[5][2] < lmList[0][2]:  # Compare y-coordinates of wrist and index knuckle
            return "Front"
        else:
            return "Backward"

    def fingersUp(self, lmList):
        fingers = []

        # Thumb
        if lmList[4][1] < lmList[3][1]:  # Compare x-coordinates of thumb tip and thumb joint
            fingers.append(1)  # Up
        else:
            fingers.append(0)  # Down

        # Fingers: Compare y-coordinates of finger tips with their corresponding joints
        for i in range(1, 5):
            if lmList[4 * i + 4][2] < lmList[4 * i + 2][2]:
                fingers.append(1)  # Up
            else:
                fingers.append(0)  # Down

        return fingers


class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        # Adjust the circle_radius to make the circles smaller
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    # Draw landmarks with smaller circles
                    self.mpDraw.draw_landmarks(
                        img, faceLms,
                        self.mpFaceMesh.FACEMESH_CONTOURS,
                        self.drawSpec, self.drawSpec
                    )
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0

    handDetector = HandDetector(maxHands=2)
    faceMeshDetector = FaceMeshDetector(maxFaces=2)

    while True:
        success, img = cap.read()
        if not success:
            break

        # Detect hands
        img = handDetector.findHands(img)
        lmList = handDetector.findPosition(img)
        if lmList:
            # Determine hand orientation
            orientation = handDetector.handOrientation(lmList)
            print(f"Hand is {orientation}")

            # Determine how many fingers are up or down
            fingers = handDetector.fingersUp(lmList)
            numFingersUp = fingers.count(1)
            numFingersDown = fingers.count(0)
            print(f"Fingers Up: {numFingersUp}, Fingers Folded: {numFingersDown}")

        # Detect face mesh
        img, faces = faceMeshDetector.findFaceMesh(img)
        if len(faces) != 0:
            print("Face landmarks detected")

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
