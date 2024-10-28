import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  # Import SVM
from sklearn.preprocessing import LabelEncoder  # To encode 'fake' and 'real' labels
from sklearn.metrics import accuracy_score  # To evaluate model performance
import nltk

# Download stopwords
nltk.download('stopwords')

# Initialize stemmer
port_stem = PorterStemmer()

class FakeNewsDetector:
    def __init__(self, dataset_path, model_type='logistic'):
        self.dataset_path = dataset_path
        self.vectorizer = TfidfVectorizer()
        self.stop_words = set(stopwords.words('english'))  # Store stopwords as a set for faster lookup
        
        # Choose model type
        if model_type == 'svm':
            self.model = SVC(probability=True)  # Use SVC with probability for better prediction
        else:
            self.model = LogisticRegression()
        
        self.load_and_preprocess_data()
    
    def load_and_preprocess_data(self):
        # Load dataset
        news_dataset = pd.read_csv(self.dataset_path)

        # Fill missing values in the 'text' column
        news_dataset['text'].fillna('', inplace=True)

        # Encode 'fake' and 'real' labels to binary values
        label_encoder = LabelEncoder()
        news_dataset['label'] = label_encoder.fit_transform(news_dataset['label'])  # 'fake' becomes 1, 'real' becomes 0

        # Stemming
        news_dataset['text'] = news_dataset['text'].apply(self.stemming)

        # Separate data and label
        self.X = news_dataset['text'].values
        self.Y = news_dataset['label'].values

        # Train-test split (70% training, 30% testing)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=0.3, stratify=self.Y, random_state=2)

        # Create DataFrames for train and test datasets based on original indices
        train_indices = np.where(np.isin(self.X, self.X_train))[0]
        test_indices = np.where(np.isin(self.X, self.X_test))[0]

        train_data = news_dataset.iloc[train_indices]
        test_data = news_dataset.iloc[test_indices]

        # Save train and test datasets to CSV
        self.save_split_data(train_data, test_data)

        # Convert text data to numerical data using TF-IDF
        self.vectorizer.fit(self.X_train)
        self.X_train = self.vectorizer.transform(self.X_train)
        self.X_test = self.vectorizer.transform(self.X_test)

        # Train the model on the training data (70%)
        self.model.fit(self.X_train, self.Y_train)


        
    def stemming(self, content):
        # Use regex to clean the content in one go
        cleaned_content = re.sub('[^a-zA-Z\s]', '', content)
        cleaned_content = cleaned_content.lower()
        words = cleaned_content.split()
        
        # Stem words while filtering out stopwords
        stemmed_content = [port_stem.stem(word) for word in words if word not in self.stop_words]
        return ' '.join(stemmed_content)

    def save_split_data(self, train_data, test_data):
        """Save the training and testing datasets as CSV files."""
        train_data.to_csv('train_news_dataset.csv', index=False)  # Save training data
        test_data.to_csv('test_news_dataset.csv', index=False)    # Save testing data
        print("Training and testing datasets have been saved as 'train_news_dataset.csv' and 'test_news_dataset.csv'.")
    
    def predict(self, text):
        # Predict whether a given text is 'Fake' or 'Real'
        processed_text = self.vectorizer.transform([self.stemming(text)])
        prediction = self.model.predict(processed_text)
        return 'Real' if prediction[0] == 1 else 'Fake'

    def evaluate(self):
        # Evaluate model performance on the test dataset (30%)
        Y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.Y_test, Y_pred)
        print(f"Model accuracy on test dataset: {accuracy * 100:.2f}%")
        return accuracy


# Example usage:
# detector = FakeNewsDetector('E:/fake_new_prediction/news_dataset.csv', model_type='svm')
# detector.evaluate()  # Evaluate the model using the test dataset only
