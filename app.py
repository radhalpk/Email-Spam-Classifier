import streamlit as st
import pickle
import os
import string
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer
from sklearn.feature_extraction.text import HashingVectorizer

# Set the NLTK data path to a local directory
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Download necessary NLTK data
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

# Initialize the stemmer and load the stopwords
ps = PorterStemmer()
sw = stopwords.words('english')

# Define the text processing function
def transfer(text):
    sp = string.punctuation
    text = text.lower()
    words = nltk.word_tokenize(text)
    filtered_words = [ps.stem(word) for word in words if word not in sw and word not in sp and word.isalnum()]
    filtered_text = [" ".join(filtered_words)]
    
    hash_vector = HashingVectorizer(n_features=10000)
    vector = hash_vector.fit_transform(filtered_text).toarray()
    return vector

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = pickle.load(open(model_path, 'rb'))

# Define the Streamlit app
def main():
    st.title("Email Classifier")
    st.write("Enter the email content below to check if it's spam or ham:")

    # Input text area
    user_input = st.text_area("Email Content")

    # Button to classify the email
    if st.button("Classify"):
        if user_input:
            # Process the input and make a prediction
            processed_input = transfer(user_input)
            prediction = model.predict(processed_input)

            if prediction == 0:
                st.success("The email is not spam, it's ham.")
            else:
                st.error("The email is spam.")
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
