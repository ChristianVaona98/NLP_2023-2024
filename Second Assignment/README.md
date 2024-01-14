# ChatGPT Slicing Algorithm

## Overview

This Python script implements an algorithm to generate slices of excessive context windows for ChatGPT 3.5. The slicing algorithm is designed to handle both input texts that exceed the standard context window size of 128 MB and not. The criteria for generating coverage include handling overlaps, ensuring non-inclusion of one slice in another, and maintaining sufficient diversity between adjacent slices.

## Technologies Used

- **nltk:** Natural Language Toolkit for natural language processing tasks.
- **sklearn:** Scikit-learn library for vectorization and cosine distance calculations.

## Functions

### `preprocess_text(text)`

This function takes an input text and performs the following preprocessing steps:

1. Tokenizes the text using nltk's `word_tokenize`.
2. Removes stopwords and applies stemming using nltk's `PorterStemmer`.
3. Returns both the processed text and the original tokenized text.

### `are_slices_different_enough(slice1, slice2, slice_threshold)`

This function compares two slices based on cosine distance after vectorization. It utilizes scikit-learn's `CountVectorizer` for vectorization and `cosine_distances` for distance calculation.

- **Parameters:**
  - `slice1` and `slice2`: Slices to be compared.
  - `slice_threshold`: Threshold for cosine distance, ensuring slices are different enough.

- **Returns:**
  - `True` if slices are different enough, `False` otherwise.

### `generate_slices(input_text, context_window_size, slice_threshold)`

This main function generates slices for the input text based on the specified criteria. It utilizes the `preprocess_text` function for text preprocessing and `are_slices_different_enough` for comparing slices.

- **Parameters:**
  - `input_text`: The input text to be sliced.
  - `context_window_size`: Standard size of the context window (default: 128 MB).
  - `slice_threshold`: Threshold for cosine distance (default: 0.20).

- **Returns:**
  - List of original slices that cover the entire input length.

## Example Usage

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# ... (Include the provided functions here)

if __name__ == "__main__":
    # Adjustable parameters
    user_context_window_size = 128  # Change as needed
    user_slice_threshold = 0.20  # Change as needed

    # Example usage:
    user_input_text = """Your input text goes here."""
    
    slices = generate_slices(user_input_text, context_window_size=user_context_window_size, slice_threshold=user_slice_threshold)
    
    for i, slice_text in enumerate(slices):
        print(f"Original Slice {i + 1}:", slice_text)
```

## Important Notes

Adjust `user_context_window_size` and `user_slice_threshold` in the __main__ block according to your requirements.
Ensure the input text is in a format suitable for natural language processing tasks.

## Dependencies
- Ensure you have the required libraries installed by running:
  ```bash
  pip install scikit-learn nltk
  ```
