from confusion_demo.models import ConfusionCounts, TeachingCase


def build_quiz_answer(case: TeachingCase, counts: ConfusionCounts) -> str:
    return (
        f"{case.model_name}: "
        f"TP={counts.tp}, FN={counts.fn}, FP={counts.fp}, TN={counts.tn}。"
        f"因为正例一共 {case.positive_samples} 个, 负例一共 {case.negative_samples} 个。"
    )
