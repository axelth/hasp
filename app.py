from random import sample
import streamlit as st
import os
import random
import librosa
from hasp.predict import SoundClassifier

if 'classifier' not in st.session_state:
    st.session_state['classifier'] = SoundClassifier()

# Page Title
st.write('# Classify Sound')


label_dict = {
    'class_0': 'air_conditioner',
    'class_1': 'car_horn',
    'class_2': 'children_playing',
    'class_3': 'dog_bark',
    'class_4': 'drilling',
    'class_5': 'engine_idling',
    'class_6': 'gun_shot',
    'class_7': 'jackhammer',
    'class_8': 'siren',
    'class_9': 'street_music'
}

# file name example is class_1_07.wav
# includes 10 files per class
# choose one at random when the button is clicked
# def load_and_predict(class_nr):
#   1. get the files for the class
#   2. find the files that are at leas as long a [samples]
#   3. choose a file at random from 2
#   3.5. play the file
#   4. extract [samples].sample points
#   5. call classifier.classify
def load_and_predict(class_no: int):
    audio_dir = f"./hasp/examples/class_{class_no}/"
    audio_file = random.choice(os.listdir(audio_dir))
    audio_path = f'{audio_dir}/{audio_file}'

    # load audiofile: specify sample rate
    y, sr = librosa.load(audio_path, sr=16000)

    # loop until we've chosen a file that is long enough
    audio_playback(audio_path)

    # find a random starting point
    pred_result = st.session_state['classifier'].classify(y)

    return pred_result

def audio_playback(path: str):
   audio_path_open = open(path, "rb")
   audio_bytes = audio_path_open.read()
   st.write("## Audio Playback")
   st.audio(audio_bytes)

# Buttons with labels, one for each class
# Click to classify an audil file
cols = st.columns(5)
for i, col in enumerate(cols):
    with col:
        st.button(label_dict[f'class_{i}'],
                  key=f'class_{i}')

cols = st.columns(5)
for i, col in enumerate(cols):
    with col:
        st.button(label_dict[f'class_{i+5}'],
                  key=f'class_{i+5}')

# Display classification result
for i in range(10):
    if st.session_state[f'class_{i}']:
        pred_result = load_and_predict(i)
        if pred_result == 1:
            st.write("## DANGER")
        else:
           st.write("## SAFE")
