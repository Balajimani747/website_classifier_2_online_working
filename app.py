import nltk

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
import re
import pickle
import requests
import streamlit as st
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

wordnet_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def text_cleaning(web_content, web_title):
    text = web_title + ' ' + web_content
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\n', '  ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = word_tokenize(text)

    cleaned_text = [wordnet_lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    filtered_sentence = " ".join(cleaned_text)
    return filtered_sentence

def main():
    st.header("Web Site Classifier")
    user_url = st.text_input("Enter The URL")

    if not user_url:
        st.warning("Please enter a URL.")


    analyze_button = st.button("Analyze")

    if analyze_button:
        try:
            response = requests.get(user_url)

            if response.status_code == 200:
                st.write("Connected to website")
                web_site_text = BeautifulSoup(response.text, 'html.parser')
                title_tag = web_site_text.title
                web_content = web_site_text.get_text()
                web_title = title_tag.string

                cleaned_text = text_cleaning(web_content, web_title)

                with open('vectorizer.pkl', 'rb') as vectorizer_file:
                    loaded_vectorizer = pickle.load(vectorizer_file)

                model_input = loaded_vectorizer.transform([cleaned_text])

                with open('classifier.pkl', 'rb') as classifier_file:
                    loaded_classifier = pickle.load(classifier_file)

                predicted_value = loaded_classifier.predict(model_input)

                st.write("Predicted Value:", predicted_value)

            else:
                st.write(f"Failed to connect to the website. Status code: {response.status_code}")

        except requests.RequestException as e:
            st.error(f"An error occurred while fetching the URL: {str(e)}")

        except Exception as ex:
            st.error(f"An unexpected error occurred: {str(ex)}")

if __name__ == "__main__":
    main()
