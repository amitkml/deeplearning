# import the necessary packages
from helpers import FACIAL_LANDMARKS_68_IDXS
from helpers import FACIAL_LANDMARKS_5_IDXS
from helpers import shape_to_np
import numpy as np
import cv2

class FaceAligner:
	def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
		desiredFaceWidth=256, desiredFaceHeight=None):
		# store the facial landmark predictor, desired output left
		# eye position, and desired output face width + height
		self.predictor = predictor
		self.desiredLeftEye = desiredLeftEye
		self.desiredFaceWidth = desiredFaceWidth
		self.desiredFaceHeight = desiredFaceHeight

		# if the desired face height is None, set it to be the
		# desired face width (normal behavior)
		if self.desiredFaceHeight is None:
			self.desiredFaceHeight = self.desiredFaceWidth

	def align(self, image, gray, rect):
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = self.predictor(gray, rect)
		shape = shape_to_np(shape)
		
		#simple hack ;)
		if (len(shape)==68):
			# extract the left and right eye (x, y)-coordinates
			(lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
			(rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
		else:
			(lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
			(rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]
			
		leftEyePts = shape[lStart:lEnd]
		rightEyePts = shape[rStart:rEnd]

		# compute the center of mass for each eye
		leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
		rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

		# compute the angle between the eye centroids
		dY = rightEyeCenter[1] - leftEyeCenter[1]
		dX = rightEyeCenter[0] - leftEyeCenter[0]
		angle = np.degrees(np.arctan2(dY, dX)) - 180

		# compute the desired right eye x-coordinate based on the
		# desired x-coordinate of the left eye
		desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

		# determine the scale of the new resulting image by taking
		# the ratio of the distance between eyes in the *current*
		# image to the ratio of distance between eyes in the
		# *desired* image
		dist = np.sqrt((dX ** 2) + (dY ** 2))
		desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
		desiredDist *= self.desiredFaceWidth
		scale = desiredDist / dist

		# compute center (x, y)-coordinates (i.e., the median point)
		# between the two eyes in the input image
		eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
			(leftEyeCenter[1] + rightEyeCenter[1]) // 2)

		# grab the rotation matrix for rotating and scaling the face
		M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

		# update the translation component of the matrix
		tX = self.desiredFaceWidth * 0.5
		tY = self.desiredFaceHeight * self.desiredLeftEye[1]
		M[0, 2] += (tX - eyesCenter[0])
		M[1, 2] += (tY - eyesCenter[1])

		# apply the affine transformation
		(w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
		output = cv2.warpAffine(image, M, (w, h),
			flags=cv2.INTER_CUBIC)

		# return the aligned face
		return output
def drawPolyline(im, landmarks, start, end, isClosed=False):
	points = []
	for i in range(start, end+1):
		point = [landmarks.part(i).x, landmarks.part(i).y]
		points.append(point)

	points = np.array(points, dtype=np.int32)
	cv2.polylines(im, [points], isClosed, (255, 200, 0),
				  thickness=2, lineType=cv2.LINE_8)

# Use this function for any model other than
# 68 points facial_landmark detector model
def renderFace2(im, landmarks, color=(0, 255, 0), radius=3):
	for p in landmarks.parts():
		cv2.circle(im, (p.x, p.y), radius, color, -1)


def writeLandmarksToFile(landmarks, landmarksFileName):
	with open(landmarksFileName, 'w') as f:
		for p in landmarks.parts():
			f.write("%s %s\n" %(int(p.x),int(p.y)))
		f.close()

def GenerateLandMarkFile(faceRects):
	import dlib
	import cv2
	import numpy as np
	# from renderFace import renderFace

	import matplotlib.pyplot as plt
	import matplotlib
	matplotlib.rcParams['figure.figsize'] = (6.0,6.0)
	matplotlib.rcParams['image.cmap'] = 'gray'


  # Loop over all detected face rectangles

	for i in range(0, len(faceRects)):
		newRect = dlib.rectangle(int(faceRects[i].left()),
								 int(faceRects[i].top()),
								 int(faceRects[i].right()),
								 int(faceRects[i].bottom()))
  # For every face rectangle, run landmarkDetector
		landmarks = landmarkDetector(im, newRect)
  # Print number of landmarks
		if i==0:
			print("Number of landmarks",len(landmarks.parts()))

  # Store landmarks for current face
		landmarksAll.append(landmarks)

  # Next, we render the outline of the face using
  # detected landmarks.
		renderFace(im, landmarks)

  # The code below saves the landmarks to
  # results/family_0.txt … results/family_4.txt.
		landmarksFileName = landmarksBasename +"_"+ str(i)+ ".txt"
		print("Saving landmarks to", landmarksFileName)
  # Write landmarks to disk
		writeLandmarksToFile(landmarks, landmarksFileName)
