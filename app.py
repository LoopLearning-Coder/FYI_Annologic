import streamlit as st
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import pickle
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import mean_squared_error


st.set_page_config(layout='wide', initial_sidebar_state='expanded')

st.title('FYI - Annologic Demo')

def main():
    with st.spinner("Unpacking model... Please wait."):
        model = pickle.load(open('popularity.pkl','rb'))

    #st.sidebar.image(imag,use_column_width=True)


    st.sidebar.header('Predict the Popularity of your Music')
    test = pd.read_csv('test.csv')
    train = pd.read_csv('train.csv')
    if st.sidebar.button('Check Model Performance'):
        train_rmse = mean_squared_error(train['popularity'],model.predict(train.drop('popularity',axis=1)),squared=False)
        test_rmse = mean_squared_error(test['popularity'],model.predict(test.drop('popularity',axis=1)),squared=False)
        col1, col2 = st.columns(2,gap='medium')
    
        col1.metric('Train RMSE:', str(round(train_rmse,3))+'%',help='Performance of model on train data')
        col2.metric('Test RMSE:', str(round(test_rmse,3))+'%',help='Performance of model on test data')
        
    
    duration = st.slider('duration in milliseconds:',0,5237295,100)
    explicit = st.selectbox('Song has explicit content?',[True,False])
    if explicit == 'True':
        explicit = 1
    else:
        explicit = 0
    genre = st.selectbox('Which Genre is your music',['acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient',
       'anime', 'black-metal', 'bluegrass', 'blues', 'brazil',
       'breakbeat', 'british', 'cantopop', 'chicago-house', 'children',
       'chill', 'classical', 'club', 'comedy', 'country', 'dance',
       'dancehall', 'death-metal', 'deep-house', 'detroit-techno',
       'disco', 'disney', 'drum-and-bass', 'dub', 'dubstep', 'edm',
       'electro', 'electronic', 'emo', 'folk', 'forro', 'french', 'funk',
       'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove',
       'grunge', 'guitar', 'happy', 'hard-rock', 'hardcore', 'hardstyle',
       'heavy-metal', 'hip-hop', 'honky-tonk', 'house', 'idm', 'indian',
       'indie-pop', 'indie', 'industrial', 'iranian', 'j-dance', 'j-idol',
       'j-pop', 'j-rock', 'jazz', 'k-pop', 'kids', 'latin', 'latino',
       'malay', 'mandopop', 'metal', 'metalcore', 'minimal-techno', 'mpb',
       'new-age', 'opera', 'pagode', 'party', 'piano', 'pop-film', 'pop',
       'power-pop', 'progressive-house', 'psych-rock', 'punk-rock',
       'punk', 'r-n-b', 'reggae', 'reggaeton', 'rock-n-roll', 'rock',
       'rockabilly', 'romance', 'sad', 'salsa', 'samba', 'sertanejo',
       'show-tunes', 'singer-songwriter', 'ska', 'sleep', 'songwriter',
       'soul', 'spanish', 'study', 'swedish', 'synth-pop', 'tango',
       'techno', 'trance', 'trip-hop', 'turkish', 'world-music'])
    
    

    pred_df = pd.DataFrame({'duration_ms':[duration],'explicit':[explicit],
                                                                 'track_genre':[genre]})
    if st.button('Predict Popularity'):
        #st.write('Predicting...')
        prediction = model.predict(pred_df)
        
        st.write(f'The predicted popularity of the music is {round(prediction,2)}\%')

if __name__ == '__main__':
    main()
