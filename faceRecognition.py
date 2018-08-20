import numpy as np
import cv2
import glob
from PIL import Image
from numpy import linalg as LA
import os.path
image_list = []

columnMatrix = np.zeros((45045, 0))

# Calculating Column matrix

for filename in glob.glob('/home/surbhi/Documents/ComputerVision/Project2/Face dataset/*.jpg'): #assuming gif
    im = Image.open(filename)
    image = cv2.imread(filename,0)
    image_list.append(im)
    print filename
    # print image.shape
    elementsArray = np.array(())
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            elementsArray = np.append(elementsArray, image[i][j])

    columnVector = np.zeros((len(elementsArray), 1))
    index = 0
    for item in elementsArray:
        columnVector[index, 0] = int(item)
        index += 1
    # print columnVector.shape
    columnMatrix = np.hstack((columnMatrix, columnVector))

#Calculating Mean Matrix

avgMatrix = np.zeros((columnMatrix.shape[0],1))
for i in range(columnMatrix.shape[0]):
    sum = 0
    for j in range(columnMatrix.shape[1]):
        sum += columnMatrix[i][j]
    avg = sum / columnMatrix.shape[1]
    avgMatrix[i][0] += avg

processingMatrix = np.zeros((columnMatrix.shape[0], columnMatrix.shape[1]))

for i in range(columnMatrix.shape[0]):
    for j in range(columnMatrix.shape[1]):
        processingMatrix[i][j] = columnMatrix[i][j] - avgMatrix[i][0]

# Calculating Co-variance matrix

processingMatrixTranspose = processingMatrix.transpose()

covarianceMatrix = np.array(())
covarianceMatrix = np.dot(processingMatrixTranspose, processingMatrix)

# computing the eigen values, eigen vectors for covariance matrix

eigenValue, eigenVectors = LA.eig((covarianceMatrix))

eigenValueMatrix = np.zeros((len(eigenValue), 1))
index = 0
for item in eigenValue:
    eigenValueMatrix[index,0] = item
    index += 1

eigenSpace = np.array(())
realEigenVectors = np.array((eigenVectors))
eigenSpace = np.dot(processingMatrix, realEigenVectors)

# Projecting training face to face space

eigenSpaceTranspose = np.transpose(eigenSpace)

faceSpace = np.array(())
faceSpace = np.dot(eigenSpaceTranspose, processingMatrix)
print faceSpace.shape

### eigenfaces recognition

# Take the testing image as input

Test_image = cv2.imread('/home/surbhi/Documents/ComputerVision/Project2/TestingDataset/subject01.happy.jpg',0)
cv2.imshow('image2', Test_image)
cv2.waitKey(0)

# Subtracting mean face from test_image

columnMatrixTest = np.zeros((45045, 0))
elementsArrayTest = np.array(())

for i in range(Test_image.shape[0]):
    for j in range(Test_image.shape[1]):
        elementsArrayTest = np.append(elementsArrayTest, Test_image[i][j])

columnVectorTest = np.zeros((len(elementsArrayTest), 1))

index = 0
for item in elementsArrayTest:
    columnVectorTest[index, 0] = int(item)
    index += 1

columnMatrixTest = np.hstack((columnMatrixTest, columnVectorTest))
columnMatrixTest = columnMatrixTest - avgMatrix                   #Creating a test image column matrix
                                                                  # and subtracting it from mean face
# Compute projection onto face space

testFaceSpace = np.array(())
testFaceSpace = np.dot(eigenSpaceTranspose, columnMatrixTest)

# Reconstructing face image

reconstructedInputFace = np.array(())
reconstructedInputFace = np.dot(eigenSpace, testFaceSpace)
print reconstructedInputFace.shape

# Compute distance between input face and its reconstruction

distanceMatrix = np.array(())
distanceMatrix = LA.norm(np.subtract(reconstructedInputFace,columnMatrixTest))
print distanceMatrix
# Compute distance between input face and training images in the face space

distance = []
for i in range(len(testFaceSpace)):
    d0 = LA.norm(np.subtract(np.transpose(testFaceSpace), np.transpose(faceSpace)[i]))
    distance.append(d0)

print distance

















