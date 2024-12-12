from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import spacy
import os
from datetime import datetime
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load SpaCy for POS tagging
nlp = spacy.load("en_core_web_sm")

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", framework="pt")

def filter_significant_words(feedback_text):
    """
    Filters significant words (adjectives, adverbs, nouns) from the input text.
    Removes generic or stopwords.
    """
    doc = nlp(feedback_text)
    significant_words = [
        token.text for token in doc
        if token.pos_ in ["ADJ", "ADV", "NOUN"] and token.text.lower() not in STOPWORDS
    ]
    return " ".join(significant_words)

def preprocess_feedback(feedback_list):
    """
    Manually adjusts specific feedback to classify as neutral based on keywords or phrases.
    """
    neutral_phrases = [
        "neither satisfied nor dissatisfied",
        "there's room for improvement"
    ]
    processed_feedback = []
    for feedback in feedback_list:
        if any(phrase in feedback.lower() for phrase in neutral_phrases):
            processed_feedback.append({'text': feedback, 'label': 'NEUTRAL', 'score': 1.0})
        else:
            processed_feedback.append({'text': feedback, 'label': None})  # To be processed by model
    return processed_feedback

def gather_feedback(feedback_data):
    """
    Combines feedback data into a single list.
    """
    return [text for feedback_list in feedback_data.values() for text in feedback_list]

def analyze_feedback(feedback_list, batch_size=10, neutral_threshold=0.6):
    """
    Analyzes feedback for sentiment and incorporates manual neutral classification.
    """
    processed_feedback = preprocess_feedback(feedback_list)
    results = []

    for feedback in processed_feedback:
        if feedback['label'] == 'NEUTRAL':  # Manually classified as neutral
            results.append({'label': 'NEUTRAL', 'score': 1.0})
        else:
            try:
                predictions = sentiment_pipeline([feedback['text']])
                prediction = predictions[0]
                if prediction['score'] < neutral_threshold:
                    results.append({'label': 'NEUTRAL', 'score': prediction['score']})
                else:
                    results.append(prediction)
            except Exception as e:
                logging.error(f"Error processing feedback: {feedback['text']} -> {e}")
                results.append({'label': 'UNKNOWN', 'score': 0})

    total_feedback = len(results)
    positive_count = sum(1 for result in results if result['label'] == 'POSITIVE')
    negative_count = sum(1 for result in results if result['label'] == 'NEGATIVE')
    neutral_count = sum(1 for result in results if result['label'] == 'NEUTRAL')

    sentiment_counts = {
        'positive': positive_count,
        'negative': negative_count,
        'neutral': neutral_count
    }

    sentiment_percentages = {
        'positive': (positive_count / total_feedback) * 100 if total_feedback > 0 else 0,
        'negative': (negative_count / total_feedback) * 100 if total_feedback > 0 else 0,
        'neutral': (neutral_count / total_feedback) * 100 if total_feedback > 0 else 0
    }

    return sentiment_counts, sentiment_percentages, results

def highlight_feedback(results, feedback_list):
    """
    Highlights feedback based on sentiment: green for positive, red for negative, and yellow for neutral.
    """
    for i, result in enumerate(results):
        color = ""
        if result['label'] == 'POSITIVE':
            color = '\033[92m'  # Green
        elif result['label'] == 'NEGATIVE':
            color = '\033[91m'  # Red
        elif result['label'] == 'NEUTRAL':
            color = '\033[93m'  # Yellow
        print(f"{color}{feedback_list[i]}\033[0m")

def generate_wordcloud(positive_feedback, output_file=None, format='png'):
    """
    Generates a word cloud from the positive feedback.
    """
    all_text = " ".join(positive_feedback)

    # Add custom stopwords
    custom_stopwords = set(STOPWORDS).union({
        "work", "team", "member", "also", "can", "from", "for", "of", "with", "to", "and", "the", "that", "this"
    })

    # Generate Word Cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=custom_stopwords
    ).generate(all_text)

    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    if output_file:
        plt.savefig(output_file, format=format)
        logging.info(f"Word cloud saved as {output_file}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        plt.savefig(f"wordcloud_{timestamp}.{format}", format=format)
        logging.info(f"Word cloud saved with timestamp {timestamp}")

def plot_sentiment_distribution(results, feedback_list, output_file='sentiment_distribution.png'):
    """
    Creates a bar chart to display feedback split by sentiment.
    """
    positive_feedback = [feedback_list[i] for i, result in enumerate(results) if result['label'] == 'POSITIVE']
    neutral_feedback = [feedback_list[i] for i, result in enumerate(results) if result['label'] == 'NEUTRAL']
    negative_feedback = [feedback_list[i] for i, result in enumerate(results) if result['label'] == 'NEGATIVE']

    categories = ['Positive', 'Neutral', 'Negative']
    counts = [len(positive_feedback), len(neutral_feedback), len(negative_feedback)]

    plt.figure(figsize=(10, 6))
    plt.bar(categories, counts, color=['green', 'yellow', 'red'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Feedbacks')
    for i, v in enumerate(counts):
        plt.text(i, v + 0.5, str(v), ha='center', fontsize=12)

    plt.savefig(output_file)
    logging.info(f"Sentiment distribution plot saved as {output_file}")

# Randomized feedback data
feedback_data = {
    'Opportunities to contribute and develop skills': [
        "I had a great opportunity to enhance my skills through the workshop.",
        "The sessions were not very engaging.",
        "I am neither satisfied nor dissatisfied with the training sessions.",
        "Skill development is ongoing, but there's room for improvement."
    ],
    'Recognition and appreciation of efforts': [
        "My work was highly appreciated in the team meeting.",
        "Recognition is not consistent across the team.",
        "I feel indifferent about the recognition initiatives.",
        "There is some acknowledgment, but it's not very impactful."
    ],
    'Well-informed about QE initiatives': [
        "The updates are regular and keep us aligned with the goals.",
        "Communication could be improved for better clarity.",
        "Neutral on the level of information provided.",
        "I sometimes feel disconnected from the bigger picture despite the updates."
    ]
}

# Combine feedback from all categories
all_feedback = gather_feedback(feedback_data)

# Analyze feedback
sentiment_counts, sentiment_percentages, results = analyze_feedback(all_feedback)

# Print results
logging.info(f"Sentiment Counts: {sentiment_counts}")
logging.info(f"Sentiment Percentages: {sentiment_percentages}")

# Highlight feedback
highlight_feedback(results, all_feedback)

# Generate word cloud
generate_wordcloud(
    [feedback for i, feedback in enumerate(all_feedback) if results[i]['label'] == 'POSITIVE'], 
    output_file='wordcloud.png'
)

# Plot sentiment distribution
plot_sentiment_distribution(results, all_feedback)