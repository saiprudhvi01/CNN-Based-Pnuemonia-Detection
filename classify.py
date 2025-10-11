import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Softmax

# Global variables for lazy loading
_models = {}

def _load_model(model_name):
    if model_name not in _models:
        _models[model_name] = load_model(f"models/{model_name}.h5", compile=False, custom_objects={'Softmax': Softmax})
    return _models[model_name]

# Load models lazily
cls_model = None
fract_model = None
brain_model = None
chest_model = None
eye_model = None
kid_model = None
skin_model = None

def get_cls_model():
    global cls_model
    if cls_model is None:
        cls_model = _load_model("all-in-one")
    return cls_model

def get_fract_model():
    global fract_model
    if fract_model is None:
        fract_model = _load_model("fracture")
    return fract_model

def get_brain_model():
    global brain_model
    if brain_model is None:
        brain_model = _load_model("brain")
    return brain_model

def get_chest_model():
    global chest_model
    if chest_model is None:
        chest_model = _load_model("chest")
    return chest_model

def get_eye_model():
    global eye_model
    if eye_model is None:
        eye_model = _load_model("eye")
    return eye_model

def get_kid_model():
    global kid_model
    if kid_model is None:
        kid_model = _load_model("kidney")
    return kid_model

def get_skin_model():
    global skin_model
    if skin_model is None:
        skin_model = _load_model("skin")
    return skin_model

def classify(img):
    model = get_cls_model()
    im = cv2.resize(img, (52, 52))
    result = model.predict(np.array([im]))
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
    model = get_fract_model()
    result = model.predict(np.array([img]))
    return ['not fractured', 'fractured'][np.argmax(result)]

def brain_net(img):
    model = get_brain_model()
    result = model.predict(np.array([img]))
    return ['pituitary', 'notumor', 'meningioma', 'glioma'][np.argmax(result)]

def chest_net(img):
    model = get_chest_model()
    result = model.predict(np.array([img]))
    return ['PNEUMONIA', 'NORMAL'][np.argmax(result)]

def eye_net(img):
    model = get_eye_model()
    result = model.predict(np.array([img]))
    return ['glaucoma', 'normal', 'diabetic_retinopathy', 'cataract'][np.argmax(result)]

def kidney_net(img):
    model = get_kid_model()
    result = model.predict(np.array([img]))
    return ['Cyst', 'Tumor', 'Stone', 'Normal'][np.argmax(result)]

def skin_net(img):
    model = get_skin_model()
    result = model.predict(np.array([img]))
    return [
        'pigmented benign keratosis', 'melanoma', 'vascular lesion',
        'actinic keratosis', 'squamous cell carcinoma', 'basal cell carcinoma',
        'seborrheic keratosis', 'dermatofibroma', 'nevus'
    ][np.argmax(result)]

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
