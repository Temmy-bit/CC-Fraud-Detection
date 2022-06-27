# %%writefile app.py
import streamlit as st
# from keras.models import load_weight
import tensorflow as tf 
import cv2 as cv
import keras
import numpy as np
import matplotlib.pyplot as plt

# from keras.models import model_from_json
# model = tf.keras.models.load_model('model(1).h5')
# my_model = keras.models.load_model(filepath = "model(1).h5")
class_names = ['1000N' '100N' '10N' '200N' '20N' '500N' '50N' '5N']
# model = load_model
loaded_model = tf.keras.models.load_model('/Users/Admin/Desktop/model')
# loaded_model

img = st.file_uploader('Load an image any Nigerian Currency: ')

if img != None:
  def load_and_pred_img(img,image_size=224):

    #Read in image
    image = cv.imread('image',img)

    #Decode the read image to tensor
    img = tf.image.decode_image(image, channel = 3)

    #Resize the image
    img = tf.image.resize(img,size=[224, 224])
    img = tf.expand_dims(img,axis=0)

    img = img/255.
    pred = loaded_model.predict(img)
    pred_class = class_names[int(tf.round(np.argmax(pred)))]

    # plt.imshow(img)
    # plt.title(f'Prediction: {pred_class}')
    # plt.axis(False);
    st.image(image)
    return pred_class

  # img = load_and_prep_img('Desktop\6-04-2022\20 Naira\IMG_4403.JPG')
  # pred = model.predict(img)[0].round(0)


  # img = load_and_prep_img('Desktop\6-04-2022\20 Naira\IMG_4403.JPG')
  # pred = model.predict(img)[0].round(0)


  print('Model successfully ran')
  # img = load_and_prep_img('Desktop\6-04-2022\20 Naira\IMG_4403.JPG')
  # pred = model.predict(img)[0].round(0)


  img_shape = (224,224)
  print('Model successfully ran')
  lay = load_and_pred_img(img,image_size = img_shape)






    
    
# %%
