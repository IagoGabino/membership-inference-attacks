"""Final 20 Newsgroups DP noise variation runner."""

from src.config import NEWS20_DP_EXPERIMENTS, NEWS20_DP_NOISE_BASE_CONFIG
from src.final_runner_utils import parse_common_args, run_dp_noise_suite


def main():
    args = parse_common_args(
        "Final 20 Newsgroups DP noise variation experiments",
        "results_final/20news_dp_overfitted",
    )
    run_dp_noise_suite(
        dataset="news20",
        base_config=NEWS20_DP_NOISE_BASE_CONFIG,
        dp_experiments=NEWS20_DP_EXPERIMENTS,
        seeds=args.seeds,
        output_dir=args.output_dir,
        results_filename="20news_dp_noise_results.csv",
        summary_filename="20news_dp_noise_summary.csv",
        figure_prefix="20news",
        smoke=args.smoke,
        device_arg=args.device,
    )


if __name__ == "__main__":
    main()
