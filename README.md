# NLP_2023-2024
# Wikipedia Text Classifier

## Overview
This Python script utilizes the Wikipedia API and the Natural Language Toolkit (NLTK) library to fetch, preprocess, and classify articles into medical and non-medical categories. The classification is performed using the Naive Bayes classifier from NLTK and preprocess the text before creating features using a Bag of Words approach. In particular the extract_features function utilizes the NLTK library to create a frequency distribution of the preprocessed words for each text.

## Technology Used
- **Programming Language:** Python
- **Libraries:**
  - `wikipediaapi`: Used to interact with the Wikipedia API.
  - `nltk`: The Natural Language Toolkit for natural language processing tasks.
  
## Functions

### `get_wikipedia_content(title)`
- **Description:** Fetches the content of a Wikipedia page with the given title.
- **Parameters:**
  - `title` (str): The title of the Wikipedia page.
- **Returns:** The text content of the Wikipedia page or `None` if the page does not exist.

### `get_category_members(category, num_pages)`
- **Description:** Fetches a specified number of articles from a given Wikipedia category.
- **Parameters:**
  - `category` (str): The name of the Wikipedia category.
  - `num_pages` (int): The number of articles to fetch.
- **Returns:** A list of article contents from the specified category.

### `preprocess_text(text)`
- **Description:** Preprocesses text by tokenizing, converting to lowercase, and removing stop words.
- **Parameters:**
  - `text` (str): The input text to be preprocessed.
- **Returns:** A list of preprocessed words.

### `extract_features(annotated_texts, category)`
- **Description:** Extracts features from annotated texts by creating frequency distributions of preprocessed words.
- **Parameters:**
  - `annotated_texts` (list): A list of annotated texts.
  - `category` (str): The category label for the texts.
- **Returns:** A list of features, where each feature is a tuple containing a frequency distribution and the category label.

### `split_dataset(features)`
- **Description:** Splits the dataset into training and testing sets.
- **Parameters:**
  - `features` (list): The list of features to be split.
- **Returns:** Two lists representing the training set and testing set.

## Main Script (`__main__` block)
1. **Fetch Articles:**
   - Fetches a specified number of medical and non-medical articles from Wikipedia for selected categories.
   - Categories: Medicine, Cardiology, Neurology, Oncology, History, Building, Geography, Mathematics.

2. **Preprocess Texts:**
   - Preprocesses the fetched texts by removing stop words.

3. **Extract Features:**
   - Creates features for medical and non-medical texts using frequency distributions of preprocessed words.

4. **Combine and Shuffle:**
   - Combines medical and non-medical features into a single dataset.
   - Shuffles the dataset to ensure randomness.

5. **Train and Test Classifier:**
   - Splits the dataset into training and testing sets.
   - Trains a Naive Bayes classifier using the training set.
   - Tests the classifier using the testing set and prints the accuracy.

6. **Save Classifier:**
   - Saves the trained classifier to a file (`naive_bayes_classifier.pickle`) for future use.

## Note
- Adjust the `num_medical_articles` and `num_non_medical_articles` variables to control the number of articles fetched for each category.
- The script introduces a delay (`time.sleep(1)`) between API requests to avoid exceeding rate limits.

## Dependencies
- Ensure you have the required libraries installed by running:
  ```bash
  pip install wikipedia-api nltk
  ```
