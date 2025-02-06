import streamlit as st 
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model=tf.keras.models.load_model("train2_pottato_disease_model.keras")
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    predictions=model.predict(input_arr)
    return np.argmax(predictions)

st.sidebar.title("Plant Disease system for sustainable agriculture")
app_mode =st.sidebar.selectbox('select page',['home','disease recognization'])

from PIL import Image
img=Image.open('potato2.jpg')
st.image(img)

if(app_mode=='home'):
    st.markdown("<h1 style='text-align: center; color: blue;'>Plant Disease system for sustainable agriculture",unsafe_allow_html=True)

elif(app_mode=='disease recognization'):
    st.header('Plant Disease system for sustainable agriculture')

test_image=st.file_uploader('choose an image:')
if(st.button('show image')):
    st.image(test_image,width=4,use_container_width=True)

if(st.button('predict')):
    st.snow()
    st.write('our prediction') 
    result_index =model_prediction(test_image) 

    class_name=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
    st.success('Model is predicting its a {}'.format(class_name[result_index]))          

