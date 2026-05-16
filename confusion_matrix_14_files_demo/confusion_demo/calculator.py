from confusion_demo.models import ConfusionCounts, TeachingCase


def build_confusion_counts(case: TeachingCase) -> ConfusionCounts:
    if case.positive_samples + case.negative_samples != case.total_samples:
        raise ValueError("正负样本数量之和必须等于总样本数。")

    if case.correct_positive_predictions > case.positive_samples:
        raise ValueError("预测正确的正例数不能超过正例总数。")

    if case.correct_negative_predictions > case.negative_samples:
        raise ValueError("预测正确的负例数不能超过负例总数。")

    tp = case.correct_positive_predictions
    tn = case.correct_negative_predictions
    fn = case.positive_samples - tp
    fp = case.negative_samples - tn

    return ConfusionCounts(
        model_name=case.model_name,
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
    )
