"""Final 20 Newsgroups top-k membership risk runner."""

from src.config import NEWS20_TOPK_EXPERIMENTS
from src.final_runner_utils import parse_common_args, run_standard_suite


def main():
    args = parse_common_args("Final 20 Newsgroups top-k MIA risk experiments", "results_final/20news_topk")
    run_standard_suite(
        dataset="news20",
        experiments=NEWS20_TOPK_EXPERIMENTS,
        seeds=args.seeds,
        output_dir=args.output_dir,
        results_filename="20news_topk_results.csv",
        summary_filename="20news_topk_best_attack_summary.csv",
        figure_prefix="20news",
        smoke=args.smoke,
        topk_plots=True,
        device_arg=args.device,
    )


if __name__ == "__main__":
    main()
