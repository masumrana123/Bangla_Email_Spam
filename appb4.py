import pandas as pd
import re
import string
from imblearn.over_sampling import SMOTE  # Import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score
import streamlit as st

# Initialize Bengali stopwords
bengali_stopwords = [
    'এটা', 'করা', 'আছে', 'আমি', 'তুমি', 'সে', 'এটি', 'এই', 'যে', 'তার', 
    'আমরা', 'তোমরা', 'ওরা', 'যারা', 'এর', 'সঙ্গে', 'একটি', 'আর', 
    'কি', 'হবে', 'যদি', 'এখন', 'অথবা', 'কেন', 'কিভাবে', 'যত'
]

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in bengali_stopwords]
    return ' '.join(tokens)

# Load and preprocess the data
data = pd.read_csv("bangla_spam.csv")
data.drop_duplicates(inplace=True)
data['v1'] = data['v1'].replace(['ham', 'spam'], ['Ham', 'Spam'])
data['v2'] = data['v2'].apply(preprocess)

mess = data['v2']
cat = data['v1']

# Split the data into training and test sets
mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2, random_state=42)

# Vectorize the text data with custom Bengali stopwords
cv = CountVectorizer(stop_words=bengali_stopwords)
features_train = cv.fit_transform(mess_train)
features_test = cv.transform(mess_test)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
features_train_resampled, cat_train_resampled = smote.fit_resample(features_train, cat_train)

# Define models
models = {
    "Naive Bayes": MultinomialNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine": SVC()
}

# Initialize variables to store the best model
best_model = None
best_model_name = ""
best_accuracy = 0
best_precision = 0

# Evaluate each model
for name, model in models.items():
    model.fit(features_train_resampled, cat_train_resampled)
    predictions = model.predict(features_test)
    
    accuracy = accuracy_score(cat_test, predictions)
    precision = precision_score(cat_test, predictions, pos_label="Spam")
    
    # Print performance metrics for the current model
    print(f"{name} Model:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    
    # Choose the model with the best combined performance of accuracy and precision
    if accuracy > best_accuracy and precision > best_precision:
        best_accuracy = accuracy
        best_precision = precision
        best_model = model
        best_model_name = name

# Print the best model
print(f"\nThe best model is: {best_model_name}")

# Prediction function
def predict(message):
    preprocessed_message = preprocess(message)
    input_message = cv.transform([preprocessed_message]).toarray()
    result = best_model.predict(input_message)
    return result[0]

# Streamlit interface
st.title('Spam Detection')
st.header('Enter Your Message Below')

# Use text_input for message input
input_mess = st.text_input('Message:')

if st.button('Validate'):
    if input_mess:
        output = predict(input_mess)
        st.write(f"The message is classified as: **{output}**")
    else:
        st.write("Please enter a message.")
