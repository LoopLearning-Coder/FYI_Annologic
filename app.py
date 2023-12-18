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
    
    page = st.sidebar.selectbox('Choose a task',['Similarity Test','Popularity Predictor'])
    #st.sidebar.image(imag,use_column_width=True)

    if page == 'Popularity Predictor':
        st.header('Predict the Popularity of your Music')
        with st.spinner("Unpacking model... Please wait."):
            model = pickle.load(open('popularity.pkl','rb'))
        
        test = pd.read_csv('test.csv')
        train = pd.read_csv('train.csv')
        
        #train_rmse = mean_squared_error(train['popularity'],model.predict(train.drop('popularity',axis=1)),squared=False)
        #test_rmse = mean_squared_error(test['popularity'],model.predict(test.drop('popularity',axis=1)),squared=False)
        #col1, col2 = st.columns(2,gap='medium')
        
        #col1.metric('Train RMSE:', str(round(train_rmse,3))+'%',help='Performance of model on train data')
        #col2.metric('Test RMSE:', str(round(test_rmse,3))+'%',help='Performance of model on test data')
            
        
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
            
            st.write(f'The predicted popularity of the music is {round(prediction[0],2)}\%')
    
    elif page == 'Similarity Test':
        st.header('Check Similarity of your Lyrics')
        import os
        import re
        import string
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from typing import List, Tuple
        
        def preprocess_text(text: str) -> str:
            """
            Preprocess the text by removing special characters and converting to lowercase.
            """
            text = text.lower()
            text = re.sub(r'\d+', '', text)  # Remove numbers
            text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
            return text
        
        def read_and_preprocess(file_path: str) -> str:
            """
            Read a text file and preprocess its content.
            """
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return preprocess_text(content)
        
        def calculate_cosine_similarity(text1: str, text2: str) -> float:
            """
            Calculate the cosine similarity between two texts.
            """
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return similarity[0][0]
        
        def check_plagiarism(new_document: str, training_documents: List[str]) -> List[Tuple[str, float]]:
            """
            Check for plagiarism between a new document and a set of training documents.
            Returns a list of tuples with document name and plagiarism score.
            """
            plagiarism_scores = []
            new_doc_content = read_and_preprocess(new_document)
        
            for doc in training_documents:
                training_doc_content = read_and_preprocess(doc)
                score = calculate_cosine_similarity(new_doc_content, training_doc_content)
                plagiarism_scores.append((doc, score))
        
            # Sort the scores in descending order
            plagiarism_scores.sort(key=lambda x: x[1], reverse=True)
            
        
            return plagiarism_scores
        
        # Get a list of files
        all_file = [file for file in os.listdir() if file.startswith('song')]
        
        
        
        
        training_docs = all_file  # Containes filenames of list of files in database
        
        
        
        if st.button('View song directory'): 
            training_docs
        
        method = st.selectbox('How do you want to enter your lyrics',['upload txt','Enter text'])
        # Textbox for user input
        if method == 'Enter text':
            user_input = st.text_area("Enter song lyrics:")
            save_to_file(user_input,'user.txt')
            new_file = 'user.txt'
        
        else:# File upload
            uploaded_file = st.file_uploader("Upload a file", type=["txt"])
        
            if uploaded_file is not None:
                # If a file is uploaded, read its content
                #with open(uploaded_file, 'r') as t
                #file_contents = save_to_file(file_contents)
                new_file = uploaded_file.name
            
                
            
            # Check for plagiarism
        if st.button("Check for similarity"):
            new_document = new_file
            plagiarism_results = check_plagiarism(new_document, training_docs)
            st.write(f'The input song is most similar to the lyrics in {plagiarism_results[0][0]} with a similarity score of {plagiarism_results[0][1]}')
        
            #if st.button(f'View {plagiarism_results[0][0]}'): 
            #   with open(plagiarism_results[0][0], 'r') as file:
            #      data = file.read()
            # st.text(data)
        def save_to_file(content,name):
            # Save content to a text file
            with open(name, "w") as file:
                file.write(content)
            st.success(f"Text saved to {name}")
if __name__ == '__main__':
    main()
