# -*- coding: utf-8 -*-
import wikipediaapi
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.classify import NaiveBayesClassifier
import random
import pickle
import time

# Set your own user agent
USER_AGENT = "WikipediaNLTK_Classifier/1.0"

def get_wikipedia_content(title):
    wiki_wiki = wikipediaapi.Wikipedia('en', headers={'User-Agent': USER_AGENT})
    page_py = wiki_wiki.page(title)

    if page_py.exists():
        return page_py.text
    else:
        print(f"Page '{title}' does not exist.")
        return None

def get_category_members(category, num_pages):
    wiki_wiki = wikipediaapi.Wikipedia('en', headers={'User-Agent': 'your-user-agent'})
    category_page = wiki_wiki.page(f"Category:{category}")

    members = list(category_page.categorymembers.values())
    selected_members = members[:num_pages]

    content_list = []
    for member in selected_members:
        title = member.title
        content = get_wikipedia_content(title)
        if content:
            content_list.append(content)

    return content_list

# Function to preprocess text by removing stop words
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    return filtered_words

# Function to extract features from annotated texts
def extract_features(annotated_texts, category):
    features = []
    for text in annotated_texts:
        preprocessed_text = preprocess_text(text)
        features.append((FreqDist(preprocessed_text), category))
    return features

# Function to split the dataset into training and testing sets
def split_dataset(features):
    split_index = int(0.8 * len(features))
    training_set = features[:split_index]
    testing_set = features[split_index:]
    return training_set, testing_set

if __name__ == "__main__":
    # Fetch n number of medical and non-medical articles in this case 100 for each category
    num_medical_articles = 100
    num_non_medical_articles = 100
    # If the total number is less than that indicated it may happen that the articles in that category are fewer than those requested

    medical_categories = ['Medicine','Cardiology', 'Neurology', 'Oncology']
    non_medical_categories = ['History', 'Building', 'Geography', 'Mathematics']

    medical_texts = []
    num_articles_in_category = 0
    for category in medical_categories:
        medical_texts.extend(get_category_members(category, num_pages=num_medical_articles))
        num_articles_in_category = len(get_category_members(category, num_pages=num_medical_articles))
        print(f"Number of {category} Articles Fetched: {num_articles_in_category}")
        time.sleep(1)  # Introduce a delay between API requests

    non_medical_texts = []
    num_articles_in_category = 0
    for category in non_medical_categories:
        non_medical_texts.extend(get_category_members(category, num_pages=num_non_medical_articles))
        num_articles_in_category = len(get_category_members(category, num_pages=num_medical_articles))
        print(f"Number of {category} Articles Fetched: {num_articles_in_category}")
        time.sleep(1)  # Introduce a delay between API requests

    '''
    ONLY FOR DEBUG

    # Print the content of fetched articles
    print("Medical Articles:")
    for i, text in enumerate(medical_texts):
        print(f"{i + 1}. {text[:50]}...")

    print("\nNon-Medical Articles:")
    for i, text in enumerate(non_medical_texts):
        print(f"{i + 1}. {text[:50]}...")

    '''

    # Extract features from annotated texts
    medical_features = extract_features(medical_texts, 'medical')
    non_medical_features = extract_features(non_medical_texts, 'non-medical')

    # Combine features and shuffle the dataset
    all_features = medical_features + non_medical_features

    # Print debug information
    print(f"\nNumber of medical features: {len(medical_features)}")
    print(f"Number of non-medical features: {len(non_medical_features)}")
    print(f"Total number of features: {len(all_features)}")

    # Check if there are features before shuffling
    if all_features:
        random.shuffle(all_features)

        # Split the dataset into training and testing sets
        training_set, testing_set = split_dataset(all_features)

        # Check if the training set has data
        if training_set:
            # Train the Naive Bayes classifier
            classifier = NaiveBayesClassifier.train(training_set)

            # Test the classifier
            accuracy = nltk.classify.accuracy(classifier, testing_set)
            print(f'Accuracy: {accuracy}')

            # Save the classifier to a file for future use
            with open('naive_bayes_classifier.pickle', 'wb') as file:
                pickle.dump(classifier, file)
        else:
            print("Training set is empty. Check your data.")
    else:
        print("No features extracted. Check your data.")
