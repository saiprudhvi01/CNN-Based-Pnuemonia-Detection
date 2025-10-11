import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Softmax

# Load models with custom objects for compatibility
cls_model = load_model("models/all-in-one.h5", compile=False, custom_objects={'Softmax': Softmax})
fract_model = load_model("models/fracture.h5", compile=False, custom_objects={'Softmax': Softmax})
brain_model = load_model("models/brain.h5", compile=False, custom_objects={'Softmax': Softmax})
chest_model = load_model("models/chest.h5", compile=False, custom_objects={'Softmax': Softmax})
eye_model = load_model("models/eye.h5", compile=False, custom_objects={'Softmax': Softmax})
kid_model = load_model("models/kidney.h5", compile=False, custom_objects={'Softmax': Softmax})
skin_model = load_model("models/skin.h5", compile=False, custom_objects={'Softmax': Softmax})

def classify(img):
    im = cv2.resize(img, (52, 52))
    result = cls_model.predict(np.array([im]))
    classification = np.argmax(result)
    
    if classification == 0:
        return "Enter a valid medical image"
    elif classification == 1:
        return bone_net(im)
    elif classification == 2:
        return brain_net(im)
    elif classification == 3:
        return eye_net(im)
    elif classification == 4:
        return kidney_net(im)
    elif classification == 5:
        return chest_net(im)
    elif classification == 6:
        return skin_net(im)

def bone_net(img):
    result = fract_model.predict(np.array([img]))
    return ['not fractured', 'fractured'][np.argmax(result)]

def brain_net(img):
    result = brain_model.predict(np.array([img]))
    return ['pituitary', 'notumor', 'meningioma', 'glioma'][np.argmax(result)]

def chest_net(img):
    result = chest_model.predict(np.array([img]))
    return ['PNEUMONIA', 'NORMAL'][np.argmax(result)]

def eye_net(img):
    result = eye_model.predict(np.array([img]))
    return ['glaucoma', 'normal', 'diabetic_retinopathy', 'cataract'][np.argmax(result)]

def kidney_net(img):
    result = kid_model.predict(np.array([img]))
    return ['Cyst', 'Tumor', 'Stone', 'Normal'][np.argmax(result)]

def skin_net(img):
    result = skin_model.predict(np.array([img]))
    return [
        'pigmented benign keratosis', 'melanoma', 'vascular lesion',
        'actinic keratosis', 'squamous cell carcinoma', 'basal cell carcinoma',
        'seborrheic keratosis', 'dermatofibroma', 'nevus'
    ][np.argmax(result)]
