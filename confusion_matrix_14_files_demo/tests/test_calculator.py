import unittest

from confusion_demo.calculator import build_confusion_counts
from confusion_demo.loader import load_case


class CalculatorTestCase(unittest.TestCase):
    def test_model_a_counts(self) -> None:
        case = load_case("data/model_a.json")
        result = build_confusion_counts(case)
        self.assertEqual((result.tp, result.fn, result.fp, result.tn), (3, 3, 0, 4))

    def test_model_b_counts(self) -> None:
        case = load_case("data/model_b.json")
        result = build_confusion_counts(case)
        self.assertEqual((result.tp, result.fn, result.fp, result.tn), (6, 0, 3, 1))


if __name__ == "__main__":
    unittest.main()
