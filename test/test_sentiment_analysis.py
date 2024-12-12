import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentiment_analysis import analyze_feedback, preprocess_feedback, gather_feedback

class TestSentimentAnalysis(unittest.TestCase):

    def test_preprocess_feedback(self):
        feedback_list = [
            "I am neither satisfied nor dissatisfied with the training sessions.",
            "Skill development is ongoing, but there's room for improvement.",
            "This is a great opportunity!"
        ]
        expected_output = [
            {'text': "I am neither satisfied nor dissatisfied with the training sessions.", 'label': 'NEUTRAL', 'score': 1.0},
            {'text': "Skill development is ongoing, but there's room for improvement.", 'label': 'NEUTRAL', 'score': 1.0},
            {'text': "This is a great opportunity!", 'label': None}
        ]
        result = preprocess_feedback(feedback_list)
        self.assertEqual(result, expected_output)

    @patch("sentiment_analysis.sentiment_pipeline")
    def test_analyze_feedback(self, mock_pipeline):
        feedback_list = [
            "I had a great opportunity to enhance my skills.",
            "The sessions were not very engaging.",
            "I feel neutral about the skill development initiatives."
        ]

        # Mock the sentiment pipeline predictions
        mock_pipeline.side_effect = lambda x: [
            {'label': 'POSITIVE', 'score': 0.95},
            {'label': 'NEGATIVE', 'score': 0.85},
            {'label': 'NEUTRAL', 'score': 0.5}
        ]

        sentiment_counts, sentiment_percentages, results = analyze_feedback(feedback_list)

        # Assertions
        self.assertEqual(sentiment_counts, {'positive': 1, 'negative': 1, 'neutral': 1})
        self.assertAlmostEqual(sentiment_percentages['positive'], 33.333, places=2)
        self.assertAlmostEqual(sentiment_percentages['negative'], 33.333, places=2)
        self.assertAlmostEqual(sentiment_percentages['neutral'], 33.333, places=2)

    def test_gather_feedback(self):
        feedback_data = {
            'Category 1': ["Feedback text 1", "Feedback text 2"],
            'Category 2': ["Feedback text 3", "Feedback text 4"]
        }
        expected_output = [
            "Feedback text 1",
            "Feedback text 2",
            "Feedback text 3",
            "Feedback text 4"
        ]
        result = gather_feedback(feedback_data)
        self.assertEqual(result, expected_output)

if __name__ == '__main__':
    unittest.main()