import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from glob import glob
model = load_model('dogbreedmodel.h5')
labels = {
    0: 'afghan_hound', 1: 'airedale', 2: 'appenzeller', 3: 'basenji', 4: 'beagle', 5: 'bernese_mountain_dog',
    6: 'blenheim_spaniel', 7: 'bluetick', 8: 'border_terrier', 9: 'boston_bull', 10: 'boxer',
    11: 'briard', 12: 'bull_mastiff', 13: 'cardigan', 14: 'chihuahua', 15: 'clumber', 16: 'collie',
    17: 'dandie_dinmont', 18: 'dingo', 19: 'english_foxhound', 20: 'english_springer', 21: 'eskimo_dog',
    22: 'french_bulldog', 23: 'german_short-haired_pointer', 24: 'golden_retriever', 25: 'great_dane',
    26: 'greater_swiss_mountain_dog', 27: 'ibizan_hound', 28: 'irish_terrier', 29: 'irish_wolfhound',
    30: 'japanese_spaniel', 31: 'kelpie', 32: 'komondor', 33: 'labrador_retriever', 34: 'leonberg',
    35: 'malamute', 36: 'maltese_dog', 37: 'miniature_pinscher', 38: 'miniature_schnauzer',
    39: 'norfolk_terrier', 40: 'norwich_terrier', 41: 'otterhound', 42: 'pekinese', 43: 'pomeranian',
    44: 'redbone', 45: 'rottweiler', 46: 'saluki', 47: 'schipperke', 48: 'scottish_deerhound',
    49: 'shetland_sheepdog', 50: 'siberian_husky', 51: 'soft-coated_wheaten_terrier', 52: 'standard_poodle',
    53: 'sussex_spaniel', 54: 'tibetan_terrier', 55: 'toy_terrier', 56: 'walker_hound', 57: 'welsh_springer_spaniel',
    58: 'whippet', 59: 'yorkshire_terrier'
}

def pipeline_model(path):
    img = image.load_img(path,target_size=(224,224))
    img = image.img_to_array(img)
    img = img/255.0
    img = np.expand_dims(img,axis=0)
    pred = model.predict(img)
    max_preds = []
    pred = pred[0]
    for i in range(5):
        name = labels[pred.argmax()]
        per = round(np.amax(pred)*100,2)
        max_preds.append([name,per])
        ele = pred.argmax()
        pred = np.delete(pred,ele)
    paths = glob('static/uploads/*')
    if len(paths)>5:
        for path in paths[:4]:
            os.remove(path)
    return max_preds