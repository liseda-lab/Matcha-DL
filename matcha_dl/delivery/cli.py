import argparse
from pathlib import Path

from matcha_dl.core.actions.alignment import AlignmentAction


def run_alignment(args):
    AlignmentAction.run(
        source_file_path=args.source_ontology_file,
        target_file_path=args.target_ontology_file,
        output_dir_path=args.output_dir,
        configs_file_path=args.config_file,
        reference_file_path=args.reference_file,
        candidates_file_path=args.candidates_file,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute the alignment between two ontologies")
    parser.add_argument(
        "--source_ontology_file",
        "-s",
        type=str,
        required=True,
        help="Please provide the path to the source ontology file",
    )
    parser.add_argument(
        "--target_ontology_file",
        "-t",
        type=str,
        required=True,
        help="Please provide the path to the target ontology file",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Please provide the path to the output directory",
    )
    parser.add_argument(
        "--reference_file",
        "-r",
        type=str,
        required=False,
        help="Please provide the path to the reference file",
    )
    parser.add_argument(
        "--candidates_file",
        "-c",
        type=str,
        required=False,
        help="Please provide the path to the candidates file",
    )
    parser.add_argument(
        "--config_file",
        "-C",
        type=str,
        required=False,
        help="Please provide the path to the yaml configuration file",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if not Path(args.source_ontology_file).exists():
        raise Exception(f"Source ontology file {args.source_ontology_file} does not exist")
    if not Path(args.target_ontology_file).exists():
        raise Exception(f"Target ontology file {args.target_ontology_file} does not exist")
    if args.reference_file and not Path(args.reference_file).exists():
        raise Exception(f"Reference file {args.reference_file} does not exist")
    if args.candidates_file and not Path(args.candidates_file).exists():
        raise Exception(f"Candidates file {args.candidates_file} does not exist")
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    if args.config_file:
        config_file = Path(args.config_file)
        if not config_file.exists():
            raise Exception(f"Configuration file {args.config_file} does not exist")

    run_alignment(args)


if __name__ == "__main__":
    main()
