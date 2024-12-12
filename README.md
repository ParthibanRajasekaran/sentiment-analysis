# Sentiment Analysis Script

## Overview

This Python script is designed for analyzing the sentiment of textual feedback and generating visual insights, such as word clouds and sentiment distribution charts. It combines the power of natural language processing (NLP) with data visualization to provide actionable insights into the tone and content of feedback data.

### Key Features:
1. **Sentiment Analysis**: Uses Hugging Face's `transformers` library to analyze feedback and categorize it as Positive, Negative, or Neutral. Incorporates manual overrides for nuanced feedback.
2. **POS Filtering**: Extracts significant words (adjectives, adverbs, nouns) using SpaCy to refine textual data.
3. **Word Cloud Generation**: Creates visually appealing word clouds from positive feedback.
4. **Sentiment Distribution Charts**: Generates bar charts to display feedback sentiment distribution.
5. **Logging**: Logs key activities and outcomes for easy debugging and tracking.

---

## Dependencies

To run the script, ensure the following Python packages are installed:

- `transformers`
- `spacy`
- `matplotlib`
- `wordcloud`
- `pandas`

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

Additionally, download the SpaCy language model:

```bash
python -m spacy download en_core_web_sm
```

---

## Usage

### 1. **Feedback Data**

The script processes feedback data structured as a dictionary with categories as keys and lists of feedback strings as values:

```python
feedback_data = {
    'Category 1': ["Feedback text 1", "Feedback text 2"],
    'Category 2': ["Feedback text 3", "Feedback text 4"],
}
```

Replace the sample feedback with your own data.

---

### 2. **Running the Script**

Execute the script using Python:

```bash
python sentiment_analysis.py
```

The script performs the following steps:
1. **Sentiment Analysis**: Analyzes the sentiment of all feedback entries.
2. **Logs Results**: Prints sentiment counts and percentages to the console.
3. **Visualizations**: Generates the following outputs:
   - **Word Cloud**: From positive feedback, saved as `wordcloud.png`.
   - **Sentiment Distribution Chart**: Bar chart of feedback sentiment, saved as `sentiment_distribution.png`.

---

### 3. **Key Outputs**

#### a. **Sentiment Analysis Results**
The sentiment counts and percentages are logged to the console. Example:

```plaintext
Sentiment Counts: {'positive': 5, 'negative': 3, 'neutral': 2}
Sentiment Percentages: {'positive': 50.0, 'negative': 30.0, 'neutral': 20.0}
```

#### b. **Word Cloud**
The script generates a word cloud image file (`wordcloud.png`) highlighting the most frequent words in positive feedback.

#### c. **Sentiment Distribution Chart**
A bar chart (`sentiment_distribution.png`) visualizing the counts of Positive, Negative, and Neutral feedback.

---

## Customization

### 1. **Batch Size**
Modify the batch size for sentiment analysis by updating the `batch_size` parameter in the `analyze_feedback` function:

```python
def analyze_feedback(feedback_list, batch_size=10):
```

### 2. **Stopwords**
Add or update custom stopwords for the word cloud in the `generate_wordcloud` function:

```python
custom_stopwords = set(STOPWORDS).union({"example", "custom", "stopword"})
```

### 3. **Output File Format**
Change the output format for the word cloud (e.g., JPEG, PNG) by updating the `format` parameter in `generate_wordcloud`:

```python
def generate_wordcloud(positive_feedback, output_file=None, format='png'):
```

### 4. **Neutral Classification**
Manually classify specific feedback as neutral by adding phrases to the `neutral_phrases` list in the `preprocess_feedback` function:

```python
neutral_phrases = ["neither satisfied nor dissatisfied", "there's room for improvement"]
```

---

## Logging

The script logs all major steps and outcomes for better traceability. Example log output:

```plaintext
INFO: Sentiment Counts: {'positive': 5, 'negative': 3, 'neutral': 2}
INFO: Sentiment Percentages: {'positive': 50.0, 'negative': 30.0, 'neutral': 20.0}
INFO: Word cloud saved as wordcloud.png
INFO: Sentiment distribution plot saved as sentiment_distribution.png
```

---

## Enhancements

To further improve the script:
1. Use a fine-tuned sentiment analysis model for your specific domain.
2. Integrate additional visualizations (e.g., pie charts for sentiment percentages).
3. Add support for larger datasets with multiprocessing or distributed computing.

---

## License

Feel free to use and modify the script for personal or professional projects. Contributions and improvements are welcome!