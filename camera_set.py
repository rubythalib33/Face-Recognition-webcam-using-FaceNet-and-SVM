import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from PIL import Image
from keras.models import load_model
import pickle

def face_embedding(model, face_array):
    face_array = face_array.astype('float32')
    mean, std = face_array.mean(), face_array.std()
    face_array = (face_array - mean)/std
    samples = np.expand_dims(face_array, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

cap = cv2.VideoCapture(0)
detector = MTCNN()
model = load_model('facenet_keras.h5')
loaded_model = pickle.load(open('trained_model.sav','rb'))
label_model = pickle.load(open('label_model.sav', 'rb'))
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(frame)
    if result != []:
        x1,y1,width,height = result[0]['box']
        x1,y1 = abs(x1), abs(y1)
        x2,y2 = x1+width, y1+height
        face=frame[y1:y2, x1:x2]

        image = Image.fromarray(face)
        image = image.resize((160,160))
        im_array = np.asarray(image)

        face_embedded=face_embedding(model, im_array)
        face_embedded = np.expand_dims(face_embedded,axis=0)

        yhat_class = loaded_model.predict(face_embedded)
        yhat_probability = loaded_model.predict_proba(face_embedded)

        class_index = yhat_class[0]
        class_probability = yhat_probability[0,class_index] * 100
        predicted_names = label_model.inverse_transform(yhat_class)

        cv2.putText(frame, predicted_names[0]+" "+str(class_probability),(x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 1)


    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
