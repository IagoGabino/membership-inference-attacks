"""Final CDC Diabetes DP noise variation runner."""

from src.config import CDC_DP_EXPERIMENTS, CDC_DP_NOISE_BASE_CONFIG
from src.final_runner_utils import parse_common_args, run_dp_noise_suite


def main():
    args = parse_common_args("Final CDC DP noise variation experiments", "results_final/cdc_dp")
    run_dp_noise_suite(
        dataset="cdc",
        base_config=CDC_DP_NOISE_BASE_CONFIG,
        dp_experiments=CDC_DP_EXPERIMENTS,
        seeds=args.seeds,
        output_dir=args.output_dir,
        results_filename="cdc_dp_noise_results.csv",
        summary_filename="cdc_dp_noise_summary.csv",
        figure_prefix="cdc",
        smoke=args.smoke,
        device_arg=args.device,
    )


if __name__ == "__main__":
    main()
