"""Final 20 Newsgroups non-DP experiment runner."""

from src.config import NEWS20_EXPERIMENTS
from src.final_runner_utils import parse_common_args, run_standard_suite


def main():
    args = parse_common_args("Final 20 Newsgroups MIA experiments", "results_final/20news")
    run_standard_suite(
        dataset="news20",
        experiments=NEWS20_EXPERIMENTS,
        seeds=args.seeds,
        output_dir=args.output_dir,
        results_filename="20news_results.csv",
        summary_filename="20news_best_attack_summary.csv",
        figure_prefix="20news",
        smoke=args.smoke,
        device_arg=args.device,
    )


if __name__ == "__main__":
    main()
