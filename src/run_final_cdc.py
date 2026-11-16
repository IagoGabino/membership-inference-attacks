"""Final CDC Diabetes non-DP experiment runner."""

from src.config import CDC_EXPERIMENTS
from src.final_runner_utils import parse_common_args, run_standard_suite


def main():
    args = parse_common_args("Final CDC Diabetes MIA experiments", "results_final/cdc")
    run_standard_suite(
        dataset="cdc",
        experiments=CDC_EXPERIMENTS,
        seeds=args.seeds,
        output_dir=args.output_dir,
        results_filename="cdc_results.csv",
        summary_filename="cdc_best_attack_summary.csv",
        figure_prefix="cdc",
        smoke=args.smoke,
        device_arg=args.device,
    )


if __name__ == "__main__":
    main()
