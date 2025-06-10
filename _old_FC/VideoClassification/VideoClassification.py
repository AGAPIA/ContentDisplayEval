from DatasetAndModelUtils import *
import ntpath
import os
import argparse
from pathlib import Path
import json


TRAIN_MODE = True
EVAL_MODE = False
INFERENCE_MODE = False

def parse_args(with_json_args: Path = None):
    parser = argparse.ArgumentParser(description="Finetune the dynabicChatbot model on a text dataset")

    parser.add_argument(
        "--do_train",
        type=bool,
        default=False,
        help="True if training mode, False otherwise",
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="MCG-NJU/videomae-base", # This is the pre-trainer Video MAE model
        help="The base model to continue fine-tuning for the video classification",
    )

    parser.add_argument(
        "--fine_tuned_suffix",
        type=str,
        help="The suffix to add to the fine-tuned model",
        required=False,
    )

    parser.add_argument(
        "--do_eval",
        type=bool,
        default=False,
        help="Evaluate the model and produce a report on the evaluation",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=25,
        required=False,
        help="Number of epochs to train the model",
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        required=False,
        help="Number of epochs to train the model",
    )

    parser.add_argument(
        "--learning_rate",
        type=int,
        default=5e-5,
        required=False,
        help="Learning rate for the model",
    )

    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        required=False,
        help="Learning rate for the model",
    )

    parser.add_argument(
        "--batch_size",
        type=float,
        default=8,
        required=False,
        help="Learning rate for the model",
    )

    args = parser.parse_args()

    # Check correctness of the arguments
    if args.do_train is True and args.do_eval is True:
        raise ValueError("Training and evaluation cannot be done at the same time")

    if args.do_train is True and args.fine_tuned_suffix is None:
        raise ValueError("Fine-tuned suffix must be provided when training. Just to be sure you don't "
                         "overwrite the base model or previously fine-tuned model")

    if args.do_eval is True and args.fine_tuned_suffix is None:
        raise ValueError("Fine-tuned suffix must be provided when evaluating. Look carefully there is one there, "
                         "include the string starting after '..-base'!")

    return args

if __name__ == "__main__":
    # Parse the arguments
    args = parse_args()
    TRAIN_MODE = args.do_train
    EVAL_MODE = args.do_eval

    # Load the dataset
    downloadedDataset: DownloadedDataset = download_dataset()
    video_classification_model = load_video_classification_model(downloadedDataset=downloadedDataset,
                                                                 load_from_pretrained=not TRAIN_MODE,
                                                                 args = args)

    # Process the dataset
    processedDataset: ProcessedDataset = process_datasets(downloadedDataset=downloadedDataset,
                                                          classifModel=video_classification_model)
    # Train the model
    if TRAIN_MODE:
        # Sample a train example to see that everything is correct
        processedDataset.test_video_load()

        train_model(video_classification_model,
                    processedDataset,
                    args=args)

    # Evaluate the model and produce a local report
    elif EVAL_MODE:

        run_inference(["FIFAdataset/val"],
                      video_classification_model,
                      processedDataset,
                      args)

