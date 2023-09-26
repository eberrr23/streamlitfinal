import streamlit as st
from PIL import Image
import tensorflow as tf
import pickle
import streamlit.components.v1 as components
import cv2
from tensorflow.keras.layers import Layer
import numpy as np
from PIL import Image



st.markdown(
         f"""
         <style>
         .stApp {{
        
             background-image: url("https://cdn.pixabay.com/photo/2021/01/01/22/12/lighthouse-5880159_1280.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
        .css-1avcm0n{{
            background: blue;

        }}

         </style>
         """,
         unsafe_allow_html=True
     )






def btn_click():
    import cv2

    # Initialize the webcam
    webcam = cv2.VideoCapture(0)  # Use index 0 for the default camera (you can change it if needed)

    # Check if the webcam opened successfully
    if not webcam.isOpened():
        print("Error: Could not open webcam.")
        exit(1)

    while True:
        try:
            # Read a frame from the webcam
            check, frame = webcam.read()
        
            # Check if the frame was read successfully
            if not check:
                print("Error: Could not read frame from the webcam.")
                break

        # Display the frame
            cv2.imshow("Capturing", frame)

        # Check for key presses
            key = cv2.waitKey(1)

            if key == ord('s'):
                # Save the current frame as 'saved_img.jpg'
                cv2.imwrite(filename='saved_img.jpg', img=frame)
                print("Image saved!")

                # Read the saved image in grayscale mode
                img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
            
                # Display the grayscale image
                cv2.imshow("Captured Image", img_new)
            
                # Wait for a moment and then close the window
                cv2.waitKey(2000)
                cv2.destroyWindow("Captured Image")

                print("Processing image...")

                # Read the saved image in color mode
                img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_COLOR)
            
                # Convert the image to grayscale
                gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            
                # Resize the image to 28x28 pixels
                img_resized = cv2.resize(gray, (28, 28))
            
                # Save the resized image as 'saved_img-final.jpg'
                cv2.imwrite(filename='saved_img-final.jpg', img=img_resized)
                
                print("Image saved!")
                
                break


            elif key == ord('q'):
                print("Turning off camera.")
                break

        except KeyboardInterrupt:
            print("Turning off camera.")
            break

    # Release the webcam and close all OpenCV windows
    webcam.release()
    cv2.destroyAllWindows()



    














#with open("designing.css") as source_des:
   # st.markdown(f"<style>{source_des.read()}</style>",unsafe_allow_html=True)

# model = tf.keras.models.load_model('satellite_military.h5')
# model = pickle.load(open('satellite_military_flutter.pkl', 'rb'))

model_path = 'https://drive.google.com/file/d/1CWcbLjdQt_tXVLMBUI3gH1tLJF_0NIez/view?usp=sharing'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

#st.set_page_config(page_title="Aircraft Detection", page_icon="🪖")


st.title("Aircraft Detection")


# Input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create a Streamlit app
st.title('TFLite Model Demo')


option1=st.radio(label="How do you want to upload the image",options=("upload from a file","upload directly from the camera","none"))
if option1=="upload from a file":   
    # Upload an image for inference
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = uploaded_image.read()
        # image = image.astype(np.float32)
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
        # Preprocess the image (you might need to adapt this based on your model)
        # For example, resize the image to the input shape expected by the model
        input_shape = input_details[0]['shape']
        # image = image / 255.0  # Normalize if necessary
        image = np.array(Image.open(uploaded_image).resize((input_shape[1], input_shape[2])))
        image = image.astype(np.float32) / 255.0

        class_labels = ['A1','A2','A3','A4','A12','A6','A7','A8','A9','A10','A11','A5','A13','A14','A15','A16','A17','A18','A19','A20']  
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], image.reshape(input_shape))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_class_index = np.argmax(output_data)

        # Get the label name for the predicted class index
        predicted_label = class_labels[predicted_class_index]

        # Display the predicted label
        st.write(f'Prediction: {predicted_label}')











elif option1=="upload directly from the camera" :
    btn_click()

    print("bozo")
    image_path="saved_img.jpg"
    image=Image.open(image_path)
   
    print("yahoooo")
    #image=saved_img1.read()
    # image = image.astype(np.float32)
    st.image(image_path, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image (you might need to adapt this based on your model)
    # For example, resize the image to the input shape expected by the model
    input_shape = input_details[0]['shape']
    # image = image / 255.0  # Normalize if necessary
    image = np.array(Image.open(image_path).resize((input_shape[1], input_shape[2])))
    image = image.astype(np.float32) / 255.0

    class_labels = ['A1','A2','A3','A4','A12','A6','A7','A8','A9','A10','A11','A5','A13','A14','A15','A16','A17','A18','A19','A20']  
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], image.reshape(input_shape))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class_index = np.argmax(output_data)

    # Get the label name for the predicted class index
    predicted_label = class_labels[predicted_class_index]

    # Display the predicted label
    st.write(f'Prediction: {predicted_label}')




