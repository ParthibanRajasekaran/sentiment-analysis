import unittest
from unittest.mock import patch
from sentiment_analysis import analyze_feedback, preprocess_feedback, gather_feedback

class TestSentimentAnalysis(unittest.TestCase):

    def test_preprocess_feedback(self):
        feedback_list = [
            "I am neither satisfied nor dissatisfied with the training sessions.",
            "Skill development is ongoing, but there's room for improvement.",
            "This is a great opportunity!",
            "I feel neutral about the skill development initiatives."
        ]
        expected_output = [
            {'text': "I am neither satisfied nor dissatisfied with the training sessions.", 'label': 'NEUTRAL', 'score': 1.0},
            {'text': "Skill development is ongoing, but there's room for improvement.", 'label': 'NEUTRAL', 'score': 1.0},
            {'text': "This is a great opportunity!", 'label': None},
            {'text': "I feel neutral about the skill development initiatives.", 'label': 'NEUTRAL', 'score': 1.0}
        ]
        result = preprocess_feedback(feedback_list)
        self.assertEqual(result, expected_output)

    @patch("sentiment_analysis.pipeline")
    def test_analyze_feedback(self, mock_pipeline):
        feedback_list = [
            "I had a great opportunity to enhance my skills.",
            "The sessions were not very engaging.",
            "I feel neutral about the skill development initiatives."
        ]

        def mock_pipeline_side_effect(texts):
            if texts == ["I had a great opportunity to enhance my skills."]:
                return [{'label': 'POSITIVE', 'score': 0.95}]
            elif texts == ["The sessions were not very engaging."]:
                return [{'label': 'NEGATIVE', 'score': 0.85}]
            else:
                return [{'label': 'UNKNOWN', 'score': 0.0}]

        mock_pipeline.side_effect = mock_pipeline_side_effect

        sentiment_counts, sentiment_percentages, results = analyze_feedback(feedback_list)

        self.assertEqual(sentiment_counts, {'positive': 1, 'negative': 1, 'neutral': 1})
        self.assertAlmostEqual(sentiment_percentages['positive'], 33.33, places=2)
        self.assertAlmostEqual(sentiment_percentages['negative'], 33.33, places=2)
        self.assertAlmostEqual(sentiment_percentages['neutral'], 33.33, places=2)

        expected_results = [
            {'label': 'POSITIVE', 'score': 0.95},
            {'label': 'NEGATIVE', 'score': 0.85},
            {'label': 'NEUTRAL', 'score': 1.0}
        ]

        for i in range(len(results)):
            self.assertEqual(results[i]['label'], expected_results[i]['label'])

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

if __name__ == "__main__":
    unittest.main()