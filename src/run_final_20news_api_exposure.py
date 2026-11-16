"""Final 20 Newsgroups API exposure/output restriction runner."""

from src.config import NEWS20_API_EXPOSURE_EXPERIMENTS, NEWS20_EXPERIMENTS
from src.final_runner_utils import parse_common_args, run_api_exposure_suite


def main():
    args = parse_common_args("Final 20 Newsgroups API exposure experiments", "results_final/20news_api")
    experiments = {
        name: NEWS20_EXPERIMENTS[name]
        for name in NEWS20_API_EXPOSURE_EXPERIMENTS
        if name in NEWS20_EXPERIMENTS
    }
    run_api_exposure_suite(
        experiments=experiments,
        seeds=args.seeds,
        output_dir=args.output_dir,
        smoke=args.smoke,
        device_arg=args.device,
    )


if __name__ == "__main__":
    main()
