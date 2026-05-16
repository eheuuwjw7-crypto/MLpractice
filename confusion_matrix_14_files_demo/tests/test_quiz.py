import unittest

from confusion_demo.calculator import build_confusion_counts
from confusion_demo.loader import load_case
from confusion_demo.quiz import build_quiz_answer


class QuizTextTestCase(unittest.TestCase):
    def test_quiz_answer_contains_a_model_result(self) -> None:
        case = load_case("data/model_a.json")
        result = build_confusion_counts(case)
        text = build_quiz_answer(case, result)
        self.assertIn("TP=3, FN=3, FP=0, TN=4", text)


if __name__ == "__main__":
    unittest.main()
