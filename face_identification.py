from numpy import load, expand_dims, asarray,savez_compressed
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from random import choice
import matplotlib.pyplot as plt
import pickle

def face_embedding(model, face_array):
    face_array = face_array.astype('float32')
    mean, std = face_array.mean(), face_array.std()
    face_array = (face_array - mean)/std
    samples = expand_dims(face_array, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

data = load('dataset-preprocessed.npz')
trainX, trainY, valX, valY = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']
print('Loaded: ', trainX.shape, trainY.shape, valX.shape, valY.shape)

#using Facenet for face embedding from keras pretrained models
model = load_model('facenet_keras.h5')
print('Loaded Model')

newTrainX = list()
for face_array in trainX:
    embedding = face_embedding(model, face_array)
    newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)

newValX = list()
for face_array in valX:
    embedding = face_embedding(model, face_array)
    newValX.append(embedding)

newValX = asarray(newValX)
print(newValX.shape)

#Face Classification
in_encoder = Normalizer(norm='l2')
newTrainX = in_encoder.transform(newTrainX)
newValX = in_encoder.transform(newValX)

out_encoder = LabelEncoder()
out_encoder.fit(trainY)
newTrainY = out_encoder.transform(trainY)
newValY = out_encoder.transform(valY)
#fit the models
model_classifier = SVC(kernel='linear', probability=True)
model_classifier.fit(newTrainX, newTrainY)

#evaluate the model
yhat_train = model_classifier.predict(newTrainX)
yhat_val = model_classifier.predict(newValX)

score_train = accuracy_score(newTrainY, yhat_train)
score_test = accuracy_score(newValY, yhat_val)

print('Accuracy: /nTrain:', score_train,'/nTest:', score_test)
pickle.dump(model_classifier, open('trained_model.sav', 'wb'))
pickle.dump(out_encoder, open('label_model.sav', 'wb'))
# Debugging Section
# selection = 7#choice([i for i in range(newValX.shape[0])])
# random_face_array = valX[selection]
# random_face_embedded = newValX[selection]
# random_face_class = newValY[selection]
# random_face_name = out_encoder.inverse_transform([random_face_class])
#
# samples = expand_dims(random_face_embedded, axis=0)
# yhat_class = model_classifier.predict(samples)
# yhat_class_probability = model_classifier.predict_proba(samples)
#
# class_index = yhat_class[0]
# class_probability = yhat_class_probability[0,class_index] * 100
# predicted_names = out_encoder.inverse_transform(yhat_class)
#
# print('Predicted: %s (%.3f) '%(predicted_names[0], class_probability))
# print('Expected: %s'%(random_face_name[0]))
#
# plt.imshow(random_face_array)
# title =  '%s (%.3f) ' %(predicted_names[0], class_probability)
# plt.title(title)
# plt.show()
