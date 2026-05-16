from confusion_demo.models import ConfusionCounts, TeachingCase


def render_case_report(case: TeachingCase, counts: ConfusionCounts) -> str:
    lines = [
        f"{case.model_name}",
        f"总样本: {case.total_samples}",
        f"正例({case.positive_label_name}): {case.positive_samples}",
        f"负例({case.negative_label_name}): {case.negative_samples}",
        f"已知: 预测对了 {case.correct_positive_predictions} 个正例, "
        f"{case.correct_negative_predictions} 个负例",
        "",
        "推导:",
        f"TP = {case.correct_positive_predictions}",
        f"FN = {case.positive_samples} - {case.correct_positive_predictions} = {counts.fn}",
        f"TN = {case.correct_negative_predictions}",
        f"FP = {case.negative_samples} - {case.correct_negative_predictions} = {counts.fp}",
        "",
        f"结论: TP={counts.tp}, FN={counts.fn}, FP={counts.fp}, TN={counts.tn}",
    ]
    return "\n".join(lines)


def render_summary_table(results: list[ConfusionCounts]) -> str:
    header = f"{'模型':<10}{'TP':<6}{'FN':<6}{'FP':<6}{'TN':<6}"
    rows = [header, "-" * len(header)]
    for item in results:
        rows.append(f"{item.model_name:<10}{item.tp:<6}{item.fn:<6}{item.fp:<6}{item.tn:<6}")
    return "\n".join(rows)
