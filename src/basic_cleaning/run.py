#!/usr/bin/env python

"""
Basic cleaning step for the NYC Airbnb dataset.
Downloads sample.csv from W&B, filters price outliers,
converts last_review to datetime, and uploads clean_sample.csv.
"""

import argparse
import logging

import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(vars(args))

    logger.info("Downloading input artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Reading input data")
    df = pd.read_csv(artifact_local_path)

    logger.info("Filtering price range")
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    logger.info("Converting last_review to datetime")
    df["last_review"] = pd.to_datetime(df["last_review"])

    # Do not add the longitude/latitude filter yet.
    # That fix is added later after release 1.0.0 fails on sample2.csv.

    logger.info("Saving cleaned data")
    df.to_csv("clean_sample.csv", index=False)

    logger.info("Uploading clean_sample.csv to W&B")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic cleaning step")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input W&B artifact to clean",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the cleaned output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the cleaned output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the cleaned output artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price to keep",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price to keep",
        required=True,
    )

    args = parser.parse_args()
    go(args)
