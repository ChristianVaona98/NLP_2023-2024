import nltk
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords and apply stemming
    filtered_words = [ps.stem(word.lower()) for word in words if word.isalnum() and word.lower() not in stop_words]

    # Join the words to form the processed text
    processed_text = ' '.join(filtered_words)

    return processed_text, ' '.join(words)  # Return both processed and original text

def are_slices_different_enough(slice1, slice2, slice_threshold):
    # Vectorize slices
    vectorizer = CountVectorizer()
    vectorizer.fit_transform([slice1, slice2])
    vector1 = vectorizer.transform([slice1]).toarray()
    vector2 = vectorizer.transform([slice2]).toarray()

    # Calculate cosine distance
    distance = cosine_distances(vector1, vector2)[0, 0]

    return distance > slice_threshold

def generate_slices(input_text, context_window_size=128, slice_threshold=0.20):
    # Check if input is below the standard size of the context window
    if len(input_text) <= context_window_size:
        return [input_text], [input_text]

    # Tokenize and preprocess input text
    processed_input, original_input = preprocess_text(input_text)

    # If input is above the standard size, create slices that cover the entire input length
    vectorizer = CountVectorizer()
    vectorizer.fit_transform([processed_input])
    input_vector = vectorizer.transform([processed_input]).toarray()

    slices = []
    original_slices = []
    start = 0

    while start < len(processed_input):
        end = start + context_window_size
        current_slice = processed_input[start:end]
        current_original_slice = original_input[start:end]

        # Check if the current slice is different enough from the previous slices
        if all(are_slices_different_enough(current_slice, prev_slice, slice_threshold) for prev_slice in slices):
            slices.append(current_slice)
            original_slices.append(current_original_slice)

        # Move the start pointer to the next non-overlapping position
        start += context_window_size // 2

    return original_slices

# Example usage:
input_text = """
The quick brown fox jumps over the lazy dog. This is a test sentence for the language model. It demonstrates how the program handles short and simple input texts. The goal is to generate meaningful slices that cover the entire input length. Each slice should be diverse enough from the others based on the specified cosine distance threshold.

Programming is the art of telling a computer what to do through a set of instructions. It involves logic, problem-solving, and creativity. Writing code allows us to create software that can automate tasks, process data, and perform various functions. Learning to program opens up a world of possibilities in the field of technology.

Artificial Intelligence is a rapidly advancing field that aims to create machines capable of intelligent behavior. Machine learning, a subset of AI, enables computers to learn from data and improve their performance over time. Large Language Models, like ChatGPT, are examples of AI applications that can understand and generate human-like text.

The internet has transformed the way we access information and communicate. It connects people globally, facilitates online collaboration, and provides a platform for sharing ideas. With the rise of social media, individuals can easily connect, share updates, and engage with a wide audience.

In the ever-evolving landscape of technology, staying informed and adapting to new developments is crucial. Continuous learning and curiosity drive innovation. As we navigate the digital age, understanding the principles of technology and its impact on society becomes increasingly important.
"""
slices = generate_slices(input_text)

for i, slice_text in enumerate(slices):
    print(f"Original Slice {i + 1}:", slice_text)
