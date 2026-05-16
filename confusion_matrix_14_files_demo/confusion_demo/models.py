from dataclasses import dataclass


@dataclass(frozen=True)
class TeachingCase:
    model_name: str
    total_samples: int
    positive_samples: int
    negative_samples: int
    correct_positive_predictions: int
    correct_negative_predictions: int
    positive_label_name: str
    negative_label_name: str


@dataclass(frozen=True)
class ConfusionCounts:
    model_name: str
    tp: int
    fn: int
    fp: int
    tn: int
