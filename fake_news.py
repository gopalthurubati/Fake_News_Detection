import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import nltk

nltk.download('stopwords')

# Initialize stemmer, vectorizer, and stopwords globally
port_stem = PorterStemmer()
vectorizer = TfidfVectorizer()
stop_words = set(stopwords.words('english'))

class FakeNewsDetector:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.load_and_preprocess_data()
        self.train_models()

    def load_and_preprocess_data(self):
        # Load dataset
        news_dataset = pd.read_csv(self.dataset_path)
        news_dataset['text'].fillna('', inplace=True)  # Handle missing values

        # Label encode the 'label' column
        label_encoder = LabelEncoder()
        news_dataset['label'] = label_encoder.fit_transform(news_dataset['label'])

        # Apply stemming to the 'text' column
        news_dataset['text'] = news_dataset['text'].apply(self.stemming)

        # Split dataset into training and test sets
        self.X = news_dataset['text'].values
        self.Y = news_dataset['label'].values
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=0.3, stratify=self.Y, random_state=2
        )

        # Create DataFrames for train and test sets using their indices
        train_indices = np.where(np.isin(self.X, self.X_train))[0]
        test_indices = np.where(np.isin(self.X, self.X_test))[0]

        train_data = news_dataset.iloc[train_indices]
        test_data = news_dataset.iloc[test_indices]

        # Save split data
        self.save_split_data(train_data, test_data)

        # Fit and transform vectorizer for training and testing sets
        self.X_train_transformed = vectorizer.fit_transform(self.X_train)
        self.X_test_transformed = vectorizer.transform(self.X_test)

    def stemming(self, content):
        # Clean and apply stemming on the content
        cleaned_content = re.sub('[^a-zA-Z\s]', '', content).lower()
        words = cleaned_content.split()
        stemmed_content = [port_stem.stem(word) for word in words if word not in stop_words]
        return ' '.join(stemmed_content)

    def train_models(self):
        # Initialize and train both Logistic Regression and SVM models
        self.logistic_model = LogisticRegression(max_iter=200)
        self.svm_model = SVC(probability=True)

        # Train the models using the transformed data
        self.logistic_model.fit(self.X_train_transformed, self.Y_train)
        self.svm_model.fit(self.X_train_transformed, self.Y_train)

    def predict(self, text, model_type='logistic'):
        # Preprocess the input text and make a prediction based on the selected model
        processed_text = vectorizer.transform([self.stemming(text)])
        if model_type == 'svm':
            prediction = self.svm_model.predict(processed_text)
        else:
            prediction = self.logistic_model.predict(processed_text)
        return 'Fake' if prediction[0] == 0 else 'Real'

    def evaluate(self, model_type='logistic'):
        # Evaluate the model's performance on the test set
        if model_type == 'svm':
            Y_pred = self.svm_model.predict(self.X_test_transformed)
        else:
            Y_pred = self.logistic_model.predict(self.X_test_transformed)
        accuracy = accuracy_score(self.Y_test, Y_pred)
        return accuracy

    def save_split_data(self, train_data, test_data):
        """Save the training and testing datasets as CSV files."""
        train_data.to_csv('train_news_dataset.csv', index=False)
        test_data.to_csv('test_news_dataset.csv', index=False)
        print("Training and testing datasets have been saved as 'train_news_dataset.csv' and 'test_news_dataset.csv'.")


