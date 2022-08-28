from random import sample
import streamlit as st
import os
import random
import librosa
from hasp.predict import SoundClassifier

# initialize classifier model
if 'classifier' not in st.session_state:
    st.session_state['classifier'] = SoundClassifier()
# initialize state variables
if 'audio_to_predict' not in st.session_state:
    st.session_state['audio_to_predict'] = None
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None
# TODO: initialize the remaining state variables

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
#   2. choose a file at random from 1
#   3. store the audio data in st.session_state
def load_audio(class_no: int) -> bool:
    '''
    Load audio data into session_state
    returns:
        True if a new file was loaded
        False if the already loaded file was selected
    '''
    audio_dir = f"./hasp/examples/class_{class_no}/"
    audio_file = random.choice(os.listdir(audio_dir))
    audio_path = f'{audio_dir}/{audio_file}'

    if audio_path not in st.session_state or st.session_state['audio_path'] != audio_path:
        st.session_state['audio_path'] = audio_path
        st.session_state['class_no'] = class_no
        # load audiofile: specify sample rate
        y, sr = librosa.load(audio_path, sr=16000)
        st.session_state['audio_to_predict'] = y
        st.session_state['audio_to_play'] = open(audio_path, 'rb').read()
        return True
    return False


def audio_playback(audio_bytes: bytes, class_no: int):
    '''
    Print currently loaded audio class and create audio playback object
    '''
    class_id_str = f'class_{class_no}'
    st.write(f"## {label_dict[class_id_str]}")
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
        loaded_new = load_audio(i)
        if loaded_new:
            st.session_state['prediction'] = None

if 'audio_to_play' in st.session_state:
    audio_playback(st.session_state['audio_to_play'],
                   st.session_state['class_no'])

if st.session_state['audio_to_predict'] is not None:
    if st.session_state.get('btn_classify', None):
        pred_result = st.session_state['classifier'].classify(
            st.session_state['audio_to_predict']
        )
        st.session_state['prediction'] = pred_result
    if st.session_state['prediction'] is None:
        st.button('Do classification', key='btn_classify')
    else:
        if st.session_state['prediction'] == 1:
            st.write("## DANGER")
        else:
            st.write("## SAFE")
