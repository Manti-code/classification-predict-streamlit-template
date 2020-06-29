"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# storing and analysis
import numpy as np
import pandas as pd
import re

# visualization
import matplotlib.pyplot as plt
import warnings
import nltk
import string
import seaborn as sns

#import text classification modules
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

#text analysers
from textblob import TextBlob
from gensim.summarization import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.stem.porter import * 
from wordcloud import WordCloud
import spacy
from spacy import displacy
from spacy.tokens import Span
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer

# import train/test split module
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# import scoring metrice
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# suppress cell warnings
warnings.filterwarnings('ignore')

#allows images to be imported into streamlit
from PIL import Image

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

def remove_pattern(input_txt, pattern):
    import re
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt

def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx,Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document,3)         
    summary_list = [str(sentence) for sentence in summary]
    result = ''.join(summary_list)
    return result                                     

def text_analyzer(my_text):
    nlp = spacy.load("en_core_web_sm")
    docx = nlp(my_text)
    tokens = [token.text for token in docx.ents]
    alldata = [('"Tokens":{},\n"Lemma":{}'.format(token.text,token.lemma_)) for token in docx.ents]
    return alldata

def entity_analyzer(my_text):
    nlp = spacy.load("en_core_web_sm")
    docx = nlp(my_text)
    tokens = [token.text for token in docx.ents]
    entity = [(entity.text,entity.label_) for entity in docx.ents]
    #entitydata = ['"Tokens":{},\n"Entities":{}'.format(tokens,entity)]
    entitydata = '"Entities":{}'.format(entity)
    #entitydata = (ent.text+' - '+ent.label_+' - '+str(spacy.explain(ents.label_)))
    return entitydata

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    #st.title("Tweet Classifer")
    #st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    

    st.sidebar.title('Navigation')
    options = ["Overview","Home","About the Data","Explore the data","Prediction on the Go"]
    selection = st.sidebar.radio("", options)

    # Building out the "Overview" page
    st.title("Climate Change Analysis")
    
    if selection == "Overview":
        st.header("Overview")
        capture_1 = Image.open("capture_1.png")
        st.image(capture_1,width=800)
        capture_2 = Image.open("capture_2.png")
        st.image(capture_2,width=800)

    # Building out the "Home" page
    if selection == "Home":
        
        st.subheader("Background")
        st.info("Many companies are built around lessening one’s environmental impact or carbon footprint. They offer products and services that are environmentally friendly and sustainable, in line with their values and ideals. They would like to determine how people perceive climate change and whether or not they believe it is a real threat. This would add to their market research efforts in gauging how their product/service may be received.")

        #pm = st.sidebar.button("Problem Statement",key="pm")
        #elif status == "Problem Statement":
        st.subheader("Problem Statement")
        st.info("Create a model that determines whether or not a person believes in climate change or not based on their tweet")
        st.subheader("Data requirements")
        st.info("The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo. The dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were collected. Each tweet is labelled as one of the following classes:")
    # Building out the "Information" page
    if selection == "About the Data":
        
        
        st.info("General Data Summaries")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")

        if st.checkbox("Preview Data"):
            
            status = st.radio(" ",("First 5 Rows","Show All Dataset"))
            if status == "First 5 Rows":
                st.dataframe(raw.head())
            else:
                st.dataframe(raw)

            if st.button("Show data summaries"): 
                st.text("Column names")
                st.write(raw.columns)
                st.text("Number of columns")
                st.write(raw.shape[1])
                st.text("Number of rows")
                st.write(raw.shape[0])
                st.text("Data types")
                st.write(raw.dtypes)
                st.text("Summary")
                st.write(raw.describe().T)

#visualising the data
    if selection == "Explore the data":
        st.info("N.B. it is recommended that the file is removed of all the noise data. For example – language stopwords (commonly used words of a language – is, am, the, of, in etc), URLs or links, social media entities (mentions, hashtags), punctuations and industry specific words. This step deals with removal of all types of noisy entities present in the text. To proceed click Preprocessing below")
        
        
        if st.button("Preprocessing"):       
            
            raw['tidy_message'] = np.vectorize(remove_pattern)(raw['message'], "@[\w]*")
            # remove special characters, numbers, punctuations
            raw['tidy_message'] = raw['tidy_message'].str.replace("[^a-zA-Z#]", " ")
            #remove short words of less than 3 letters in length
            raw['tidy_message'] = raw['tidy_message'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

    
        st.subheader("Visualising the data")
        col_names = raw.columns.tolist()
        plot_type = st.selectbox("select plot",["bar","hist","box","kde"])
        select_col_names = st.multiselect("select columns to plot",col_names)

        if st.button("Generate plot"):
            st.success("{} plot for {}".format(plot_type,select_col_names))
            if plot_type == "bar":                    
                s_grp = raw.groupby(["sentiment"]).count()
                st.bar_chart(s_grp)
                st.pyplot()
            #elif plot_type == 'area':                    
             #   plot_data = raw[select_col_names]
              #  st.area_chart(plot_data)    
            elif plot_type == 'hist':                    
                plot_data = raw[select_col_names]
                st.bar_chart(plot_data)
            elif plot_type:
                cust_plot = raw[select_col_names].plot(kind=plot_type)              
                st.write(cust_plot)
                st.pyplot()          

        st.subheader("Visuals of common words used in the tweets")
        st.markdown("The most frequent words appear in large size and the less frequent words appear in smaller sizes")

        cpw = st.checkbox("Common Positive Words",key="cpw")
        #cpw1 = st.text("Positive words: global warming, climate change, believe climate, change real")
        if cpw:

            positive_words =' '.join([text for text in raw['tidy_message'][raw['sentiment'] == 1]])
            positive_words_cloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(positive_words)
            plt.axis('off')
            plt.imshow(positive_words_cloud, interpolation="bilinear")
            plt.show()
            st.pyplot()
            #st.checkbox("Show/Hide")

        cnw = st.checkbox("Common negative Words",key="cnw")
        if cnw:

            negative_words = ' '.join([text for text in raw['tidy_message'][raw['sentiment'] == -1]])
            negative_words_cloud = WordCloud(width=800, height=500,random_state=21, max_font_size=110).generate(negative_words)
            plt.axis('off')
            plt.imshow(negative_words_cloud, interpolation="bilinear")
            plt.show()
            st.pyplot()
            #st.checkbox("Show/Hide")
        cnnw = st.checkbox("Common neutral/normal Words",key="cnnw")
        if cnnw:

            normal_words =' '.join([text for text in raw['tidy_message'][raw['sentiment'] == 0]])
            normal_words_wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)

            plt.axis('off')
            plt.imshow(normal_words_wordcloud, interpolation="bilinear")
            plt.show()
            st.pyplot()   

        cnnww = st.checkbox("Common News Words",key="cnnww")
        if cnnww:

            news_words =' '.join([text for text in raw['tidy_message'][raw['sentiment'] == 2]])
            news_words_wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(news_words)

            plt.axis('off')
            plt.imshow(news_words_wordcloud, interpolation="bilinear")
            plt.show()
            st.pyplot()

        
# Building out the predication page
    if selection == "Prediction on the Go":
        st.sidebar.success("The App allows only text to be entered. It will show any present entities,provides sentiments analysis and classifies the text as per the table on top. Enter the text in the text area provided and select the buttons of your choice below the text area")
        st.info("Prediction with ML Models")
        st.markdown("The table below shows the description of the sentiments")
        img = Image.open("class.png")
        st.image(img)
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text","Type Here")
        #named entity
    
        if st.checkbox("Show Entities"):
            #if st.subheader("Extract entities from your text"):
             #   ner= st.text_area("Enter your here","Type here",key="ner")
             #   message = ner
              #  if st.button("Extract"):
            nlp_result = entity_analyzer(tweet_text)
            st.write(nlp_result)
                    #st.write=entity_analyzer(entity)

        #sentiment analysis
        if st.checkbox("Show Sentiment Analysis"):
            
           # if st.subheader("Sentiment of your Text"):
           #     sa= st.text_area("Enter your here","Type here",key="sa")
           #     message = sa
           #     if st.button("Analyse"):

            sid = SentimentIntensityAnalyzer()
            res_sentiment=sid.polarity_scores(tweet_text)
            st.json(res_sentiment)

            if res_sentiment['compound'] == 0:
                st.write("The sentiment of your text is NEUTRAL")
            elif res_sentiment['compound'] > 0:
                st.success("The sentiment of your text is POSITIVE")
            else:
                st.warning("The sentiment of your text is NEGATIVE")

        news_vectorizer = open("resources/tfidfvect.pkl","rb")
        tweet_cv = joblib.load(news_vectorizer)
        if st.checkbox("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
            prediction = predictor.predict(vect_text)

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as Class: {}".format(prediction))
            
            Classifier = st.selectbox("Choose Classifier",['Linear SVC','Logistic regression'])

            if st.button("Classify"):
                # Transforming user input with vectorizer

                # Load your .pkl file with the model of your choice + make predictions
                # Try loading in multiple models to give the user a choice
                if Classifier =='Linear SVC':
                        st.text("Using Linear SVC classifier ..")
                        # Vectorizer
                        news_vectorizer = open("resources/vectoriser.pkl","rb")
                        tweet_cv = joblib.load(news_vectorizer)
                        predictor = joblib.load(open(os.path.join("resources/linearSVC.pkl"),"rb"))
                elif Classifier == 'Logistic regression':
                        st.text("Using Logistic Regression Classifeir ..")
                        # Vectorizer
                        news_vectorizer = open("resources/tfidfvect.pkl","rb")
                        tweet_cv = joblib.load(news_vectorizer)
                        predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))

                    results = []
                    n = 0
                    while n < len(tweet_text):
                        vect_text = tweet_cv.transform([tweet_text['message'][n]]).toarray()
                        prediction = predictor.predict(vect_text)
                        results.append((tweet_text['message'][n],prediction))
                        n+=1

                    df = pd.DataFrame(results,columns=['Message','Sentiment'])


                    #Table that tabulates the results
                    predictions = st.table(df.head(size))

                    st.success("Text Categorized as Class: {}".format(predictions))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    main()
