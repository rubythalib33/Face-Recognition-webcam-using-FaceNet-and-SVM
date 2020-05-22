from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray
from os import listdir
from os.path import isdir
import matplotlib.pyplot as plt
from numpy import savez_compressed

def extract_face(file, req_size=(160,160)):
    #Load the image
    image = Image.open(file)
    image = image.convert('RGB')
    im_array = asarray(image)
    #Using the MTCNN face detector
    detector = MTCNN()
    result = detector.detect_faces(im_array)
    #declaring bounding box
    x1,y1,width,height = result[0]['box']
    x1,y1 = abs(x1), abs(y1)
    x2,y2 = x1+width, y1+height

    face=im_array[y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = image.resize(req_size)
    im_array_result = asarray(image)
    return im_array_result

def load_faces(directory):
    faces = list()
    for filename in listdir(directory):
        path = directory + filename
        face = extract_face(path)
        faces.append(face)
    return faces

def load_dataset(directory):
    x, y = list(), list()
    for subdir in listdir(directory):
        path = directory+subdir+'/'
        if not isdir(path):
            continue
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        print('loaded {0} examples for class: {1}'.format(len(faces), subdir))
        x.extend(faces)
        y.extend(labels)
    return asarray(x), asarray(y)

trainX, trainY = load_dataset('./dataset/train/')
print(trainX.shape, trainY.shape)
valX, valY = load_dataset('./dataset/val/')
print(valX.shape, valY.shape)

savez_compressed('dataset-preprocessed.npz', trainX,trainY, valX, valY)
