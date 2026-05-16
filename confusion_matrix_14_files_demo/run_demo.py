from confusion_demo.calculator import build_confusion_counts
from confusion_demo.loader import load_case
from confusion_demo.quiz import build_quiz_answer
from confusion_demo.report import render_case_report, render_summary_table


def main() -> None:
    case_a = load_case("data/model_a.json")
    case_b = load_case("data/model_b.json")

    result_a = build_confusion_counts(case_a)
    result_b = build_confusion_counts(case_b)

    print("混淆矩阵教学示例")
    print("=" * 48)
    print(render_case_report(case_a, result_a))
    print("-" * 48)
    print(render_case_report(case_b, result_b))
    print("-" * 48)
    print("对比汇总")
    print(render_summary_table([result_a, result_b]))
    print("-" * 48)
    print("课堂速答")
    print(build_quiz_answer(case_a, result_a))
    print()
    print(build_quiz_answer(case_b, result_b))


if __name__ == "__main__":
    main()
