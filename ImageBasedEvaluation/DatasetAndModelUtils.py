from copy import deepcopy

from huggingface_hub import hf_hub_download
from transformers import TrainingArguments, Trainer, TrainerCallback
import pathlib
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import os
import imageio
import numpy as np
from IPython.display import Image
import evaluate
import torch
from pytorchvideo.data import LabeledVideoDataset
from typing import Dict, Tuple, Union, Any, List
from tkinter import *
import torchvision
import torch.nn.functional as F
from colorama import Fore, Back, Style
from datetime import datetime

from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from pydantic import BaseModel as PydanticBaseModel
from enum import Enum

# Set the seed
seed = 42
torch.manual_seed(seed)
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)

class DatasetType(Enum):
    DATASET_UCF101 = 0,
    DATASET_FIFA = 1,

DATASETUSED = DatasetType.DATASET_FIFA #DatasetType.DATASET_UCF101

class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


# This is a util class to process a dataset by a model's specifications
class ProcessedDataset(BaseModel):
    train: LabeledVideoDataset
    test: LabeledVideoDataset
    val: LabeledVideoDataset
    val_transform: torchvision.transforms.Compose

    id2label: Dict[int, str]
    label2id: Dict[str, int]

    img_mean: List[float]
    img_std: List[float]
    img_resize_to: Tuple[float, float]
    sample_rate: int
    clip_duration: float
    fps: int
    num_frames_to_sample: int


    def investigate_video(self, sample_video):
        """Utility to investigate the keys present in a single video sample."""
        for k in sample_video:
            if k == "video":
                print(k, sample_video["video"].shape)
            else:
                print(k, sample_video[k])

        print(f"Video label: {self.id2label[sample_video[k]]}")

    def unnormalize_img(self, img):
        """Un-normalizes the image pixels."""
        img = (img * self.img_std) + self.img_mean
        img = (img * 255).astype("uint8")
        return img.clip(0, 255)

    def collate_fn(self, examples):
        """The collation function to be used by `Trainer` to prepare data batches."""
        # permute to (num_frames, num_channels, height, width)
        pixel_values = torch.stack(
            [example["video"].permute(1, 0, 2, 3) for example in examples]
        )
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    def create_gif(self, video_tensor, filename="sample.gif"):
        """Prepares a GIF from a video tensor.

        The video tensor is expected to have the following shape:
        (num_frames, num_channels, height, width).
        """
        frames = []
        for video_frame in video_tensor:
            frame_unnormalized = self.unnormalize_img(video_frame.permute(1, 2, 0).numpy())
            frames.append(frame_unnormalized)
        kargs = {"duration": 0.25}
        imageio.mimsave(filename, frames, "GIF", **kargs)
        return filename

    def display_gif(self, video_tensor, gif_name="sample.gif", onlyCreate=True):
        """Prepares and displays a GIF from a video tensor."""
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        gif_filename = self.create_gif(video_tensor, gif_name)

        return Image(filename=gif_filename, imgtype='GIF') if onlyCreate is False else None


    def test_video_load(self):
        sample_video = next(iter(self.train))
        print(f"Keys from sample train video {sample_video.keys()}")

        self.investigate_video(sample_video)

        video_tensor = sample_video["video"]

        import IPython
        self.display_gif(video_tensor, onlyCreate=True)



class DownloadedDataset(BaseModel):
    dataset_root_path: Union[str, pathlib.Path]
    label2id: Dict[str, int]
    id2label: Dict[int, str]

# This is the way we encode the dataset currently
def get_class_for_filepath(filepath: pathlib.Path) -> str:
    if not isinstance(filepath, pathlib.Path):
        filepath = pathlib.Path(filepath)

    classType = None
    if len(filepath.parts) > 1:
        classType = filepath.parts[-2]
    elif "\\" in filepath.parts[0]:
        classType = filepath.parts[0].split("\\")[-2]
    elif "/" in filepath.parts[0]:
        classType = filepath.parts[0].split("/")[-2]
    return classType

# TODO generalize
class VideoClassificationModel():
    # pre-trained model from which to fine-tune
    modelCkpt: str

    # batch size for training and evaluation
    batch_size: int

    # Fine-tuned model names and paths
    modelName: str

    # Fine-tuned model
    finetuned_model_name: str

    # image processor for the model
    image_processor: VideoMAEImageProcessor

    # The model itself
    model: VideoMAEForVideoClassification

    def setup(self, modelCkpt: str,
              model_fine_tuned_suffix: str,
              batch_size: int):
        self.modelCkpt = modelCkpt
        self.batch_size = batch_size
        self.modelName = modelCkpt.split("/")[-1]

        self.finetuned_model_name = f"{self.modelName}{model_fine_tuned_suffix}"


def download_dataset() -> DownloadedDataset:
    # This is the original dataset used for training the base model
    if DATASETUSED == DatasetType.DATASET_UCF101:
        hf_dataset_identifier = "sayakpaul/ucf101-subset"
        filename = "UCF101_subset.tar.gz"
        file_path = hf_hub_download(
            repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset"
        )

        dataset_root_path = "UCF101_subset"

        dataset_root_path = pathlib.Path(dataset_root_path)
        video_count_train = len(list(dataset_root_path.glob("train/*/*.avi")))
        video_count_val = len(list(dataset_root_path.glob("val/*/*.avi")))
        video_count_test = len(list(dataset_root_path.glob("test/*/*.avi")))
        video_total = video_count_train + video_count_val + video_count_test
        print(f"Total videos: {video_total}")

        all_video_file_paths = (
                list(dataset_root_path.glob("train/*/*.avi"))
                + list(dataset_root_path.glob("val/*/*.avi"))
                + list(dataset_root_path.glob("test/*/*.avi"))
        )

    # Fifa fine-tuning dataset
    elif DATASETUSED == DatasetType.DATASET_FIFA:
        dataset_root_path = "FIFAdataset"

        dataset_root_path = pathlib.Path(dataset_root_path)
        video_count_train = len(list(dataset_root_path.glob("train/*/*.mp4")))
        video_count_val = len(list(dataset_root_path.glob("val/*/*.mp4")))
        video_count_test = len(list(dataset_root_path.glob("test/*/*.mp4")))
        video_total = video_count_train + video_count_val + video_count_test
        print(f"Total videos: {video_total}")

        all_video_file_paths = (
                list(dataset_root_path.glob("train/*/*.mp4"))
                + list(dataset_root_path.glob("val/*/*.mp4"))
                + list(dataset_root_path.glob("test/*/*.mp4"))
        )
        
    else:
        assert False, f"Unknown dataset type: {DATASETUSED}"



    class_labels = sorted({get_class_for_filepath(path) for path in all_video_file_paths}) #sorted({str(path).split("/")[2] for path in all_video_file_paths})
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}

    downloadedDataset = DownloadedDataset(id2label=id2label,
                                          label2id=label2id,
                                          dataset_root_path=dataset_root_path
                                          )
    return downloadedDataset

def process_datasets(downloadedDataset: DownloadedDataset,
                     classifModel: VideoClassificationModel) -> ProcessedDataset:
    print(f"Unique classes: {list(downloadedDataset.label2id.keys())}.")

    import pytorchvideo.data

    from pytorchvideo.transforms import (
        ApplyTransformToKey,
        Normalize,
        RandomShortSideScale,
        RemoveKey,
        ShortSideScale,
        UniformTemporalSubsample,
    )

    from torchvision.transforms import (
        Compose,
        Lambda,
        RandomCrop,
        RandomHorizontalFlip,
        Resize,
    )

    img_mean = classifModel.image_processor.image_mean
    img_std = classifModel.image_processor.image_std
    if "shortest_edge" in classifModel.image_processor.size:
        img_height = img_width = classifModel.image_processor.size["shortest_edge"]
    else:
        img_height = classifModel.image_processor.size["height"]
        img_width = classifModel.image_processor.size["width"]
    img_resize_to = (img_height, img_width)

    num_frames_to_sample = classifModel.model.config.num_frames
    sample_rate = 4
    fps = 30
    clip_duration = num_frames_to_sample * sample_rate / fps


    # Training dataset transformations.
    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(img_mean, img_std),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(img_resize_to),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )

    # Training dataset.
    train_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(downloadedDataset.dataset_root_path, "train"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )

    # Validation and evaluation datasets' transformations.
    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(img_mean, img_std),
                        Resize(img_resize_to, antialias=True),
                    ]
                ),
            ),
        ]
    )

    # Validation and evaluation datasets.
    val_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(downloadedDataset.dataset_root_path, "val"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )

    test_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(downloadedDataset.dataset_root_path, "test"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )

    print(f"Num videos in dataset- training:{train_dataset.num_videos}, val:{val_dataset.num_videos}, "
          f"test:{test_dataset.num_videos}")

    processedDataset = ProcessedDataset(train=train_dataset,
                                      test=test_dataset,
                                      val=val_dataset,
                                        val_transform=val_transform,
                                      id2label=downloadedDataset.id2label,
                                      label2id=downloadedDataset.label2id,
                                      img_mean=img_mean,
                                      img_std=img_std,
                                      img_resize_to=img_resize_to,
                                      sample_rate=sample_rate,
                                      clip_duration=clip_duration,
                                      fps=fps,
                                      num_frames_to_sample=num_frames_to_sample)



    return processedDataset

# We can access the `num_videos` argument to know the number of videos we have in the
# dataset.


def load_video_classification_model(downloadedDataset: DownloadedDataset,
                                    load_from_pretrained,
                                    args) -> VideoClassificationModel:
    vcm = VideoClassificationModel()
    vcm.setup(modelCkpt=args.base_model,
            batch_size=args.batch_size,
            model_fine_tuned_suffix=args.fine_tuned_suffix)


    # Using the same image processor in both cases
    vcm.image_processor = VideoMAEImageProcessor.from_pretrained(vcm.modelCkpt)

    if load_from_pretrained:
        vcm.model = VideoMAEForVideoClassification.from_pretrained(vcm.finetuned_model_name)
    else:
        vcm.model = VideoMAEForVideoClassification.from_pretrained(
            vcm.modelCkpt,
            label2id=downloadedDataset.label2id,
            id2label=downloadedDataset.id2label,
            ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )

    return vcm

def create_trainer(video_classification_model: VideoClassificationModel,
                   processed_dataset: ProcessedDataset,
                   args: Any) -> Trainer:

    metric = evaluate.load("accuracy")

    # the compute_metrics function takes a Named Tuple as input:
    # predictions, which are the logits of the model as Numpy arrays,
    # and label_ids, which are the ground-truth labels as Numpy arrays.
    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions."""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    class CustomCallback(TrainerCallback):

        def __init__(self, trainer) -> None:
            super().__init__()
            self._trainer = trainer

        def on_epoch_end(self, args, state, control, **kwargs):
            if control.should_evaluate:
                control_copy = deepcopy(control)
                self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
                return control_copy

    max_steps = (processed_dataset.train.num_videos // video_classification_model.batch_size) * args.num_epochs

    args = TrainingArguments(
        video_classification_model.finetuned_model_name,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=min(max_steps, max(max_steps/4, 500)),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=video_classification_model.batch_size,
        per_device_eval_batch_size=video_classification_model.batch_size,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        #load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        max_steps=max_steps,
    )

    trainer = Trainer(
        video_classification_model.model,
        args,
        train_dataset=processed_dataset.train,
        eval_dataset=processed_dataset.val,
        tokenizer=video_classification_model.image_processor,
        compute_metrics=compute_metrics,
        data_collator=processed_dataset.collate_fn,
    )
    trainer.add_callback(CustomCallback(trainer))

    return trainer


# Train the model
def train_model(video_classification_model: VideoClassificationModel,
                processed_dataset: ProcessedDataset,
                args):

    trainer = create_trainer(video_classification_model, processed_dataset, args)

    # Train !
    train_results = trainer.train()

    # Save model and store metrics !
    trainer.save_model()
    test_results = trainer.evaluate(processed_dataset.test)
    trainer.log_metrics("test", test_results)
    trainer.save_metrics("test", test_results)
    trainer.save_state()

# Evaluate the mode
def evaluate_model(video_classification_model: VideoClassificationModel,
                   processed_dataset: ProcessedDataset,
                   list_of_paths_to_evaluate_manually : List[pathlib.Path],
                   args):
    # Part 1: cases by case evaluation and write a report to a file


    for video_index, video_sample in enumerate(processed_dataset.val):
        pixel_values = video_sample["video"].unsqueeze(0)
        label = video_sample["label"]

        outputs = video_classification_model.model(pixel_values)


    trainer = create_trainer(video_classification_model, processed_dataset, args)

    # Evaluate !
    test_results = trainer.evaluate(processed_dataset.val)

    eval_timestamp = datetime.now().strftime("%Y%_m_%d_%H%M_%S")
    trainer.log_metrics(f"results/eval_results{eval_timestamp}.txt", test_results)


# This function takes a video path and transforms it in the similar way the dataset is processed for the model
# Note that the "test" dataset kind of transforms are used
def get_video_in_model_format(video_path, datasetInstance: ProcessedDataset):
    # Loaded the encoded video first using the dataset loader
    label_str = get_class_for_filepath(video_path)
    info_dict = {'label': datasetInstance.label2id[label_str]}
    videodata = datasetInstance.test.video_path_handler.video_from_path(
        video_path,
        decode_audio=datasetInstance.test._decode_audio,
        decoder=datasetInstance.test._decoder)

    # Sample frames from the dataset
    (
        clip_start,
        clip_end,
        clip_index,
        aug_index,
        is_last_clip,
    ) = datasetInstance.test._clip_sampler(0, videodata.duration, info_dict)

    # Get frames and fill in the output dict
    loaded_clip = videodata.get_clip(clip_start, clip_end)
    frames = loaded_clip["video"]
    audio_samples = loaded_clip["audio"]
    videoInModelFormat = {
        "video": frames,
        "video_name": videodata.name,
        "video_index": 0,
        "aug_index": 0,
        **info_dict,
        **({"audio": audio_samples} if audio_samples is not None else {}),
    }

    # Transform the model using the same transformations used by the dataset
    datasetInstance.test._transform(videoInModelFormat)

    return videoInModelFormat


# Run inference given a video sample in the model format, the model loaded, and a processed dataset to find out
# the keys and other stuff
def run_inference(videoPaths: Union[List[pathlib.PurePath], List[str]],
                  modelClassif: VideoClassificationModel,
                  processedDataset: ProcessedDataset,
                  args,
                  show_output: bool = False,
                  show_plots_too: bool = False):

    # Gather all the videos that need to be tested
    files_to_test: List[str] = []
    for video_path in videoPaths:
        for root, dirs, files in os.walk(video_path):
            files_to_test.extend([os.path.join(root, fi) for fi in files if fi.endswith((".mp4", ".avi"))])

    # Eliminate duplicates
    files_to_test = list(set(files_to_test))

    # Prepare the statistics dataframe
    import pandas as pd
    from datetime import datetime
    df_results = pd.DataFrame(columns=["correct", "label", "predicted", "video"])
    df_results = df_results.astype(dtype={"correct": "bool", "label": "int8", "predicted": "int8", "video": "str"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = modelClassif.model.to(device)
    batch_pixel_values = []
    batch_labels = []
    batch_paths = []
    num_files_to_eval = len(files_to_test)

    # For each video, get the video in the model format and run inference
    num_in_curr_batch = 0

    for videoIndex in range(num_files_to_eval):
        videoPath = files_to_test[videoIndex]
        batch_paths.append(videoPath) # Safest way :)

        # Get video in the model format
        videoSample = get_video_in_model_format(videoPath, processedDataset)

        # (num_frames, num_channels, height, width)
        video_data = videoSample['video']
        video_label = videoSample['label']
        permuted_sample_test_video = video_data.permute(1, 0, 2, 3)

        batch_pixel_values.append(permuted_sample_test_video)
        batch_labels.append(torch.tensor([video_label]))
        num_in_curr_batch += 1

        # If last or matched batchsize, send to model and get outputs
        if num_in_curr_batch >= args.batch_size or videoIndex == num_files_to_eval - 1:
            lables_tensor = torch.cat(batch_labels, dim=-1)
            inputs = {
                "pixel_values": torch.stack(batch_pixel_values),
                "labels": lables_tensor # this can be skipped if you don't have labels available.
            }

            inputs = {k: v.to(device) for k, v in inputs.items()}

            # forward pass
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

                # Stabilize computations
                logits -= logits.min() # bring lower range to 0
                logits /= logits.max() # bring the upper range to 1
                logits = F.normalize(logits, p=2, dim=-1)

            predictions = outputs.logits.argmax(-1).to('cpu')
            video_paths = files_to_test[(videoIndex - num_in_curr_batch  + 1): (videoIndex + 1)]
            correct_labels = torch.eq(lables_tensor, predictions)

            batch_results = []
            items_added = 0
            for path, label, prediction, is_correct in zip(batch_paths, lables_tensor, predictions, correct_labels):
                df_results = df_results._append({"correct": is_correct.item(),
                                      "label": label.item(),
                                      "predicted": prediction.item(),
                                        "video": path,
                                      },
                                                ignore_index=True)
                items_added += 1

            assert items_added <= args.batch_size, f"Items added {items_added} is more than batch size {args.batch_size}"

            #df_results = df_results.loc[ len(df_results) ]

            num_in_curr_batch = 0
            batch_pixel_values = []
            batch_labels = []
            batch_paths = []

    eval_timestamp = datetime.now().strftime("%Y%_m_%d_%H%M_%S")
    df_results.to_csv(f"results/eval_results{eval_timestamp}.csv")

    """
        # Output stuff
        if show_output:
            assert "Needs more love and refactoring"
            if show_plots_too:
                import matplotlib.pyplot as plt

            for videoIndex, videoSample in enumerate(videoSamples):
                logits_for_this_video = logits[videoIndex]
                output_probs = logits_for_this_video.tolist()

                print("-------")
                print(f"File name {videoPaths[videoIndex]} results:")
                logits_as_str = {(f"{processedDataset.id2label[clsid]}|{clsid}"): clsprob for clsid, clsprob in
                                 enumerate(output_probs)}
                print(f"Output probabilities: {logits_as_str}")
                predicted_class_idx = logits_for_this_video.argmax(-1).item()

                correct_class = get_class_for_filepath(videoPaths[videoIndex])
                predicted_class = processedDataset.id2label[predicted_class_idx]
                is_correct_class = correct_class == predicted_class

                print((
                          Fore.GREEN if is_correct_class else Fore.RED) + f"{'CORRECT' if correct_class == predicted_class else 'WRONG'} Predicted: {predicted_class}.", )
                print(Style.RESET_ALL)

                if show_plots_too:
                    classesStr = list(logits_as_str.keys())
                    classesProbs = list(logits_as_str.values())
                    fig = plt.figure(figsize=(10, 5), num=f"{videoSample['video_name']}")
                    ax = fig.subplots(1, 1)
                    ax.bar(classesStr, classesProbs, width=0.5, align='center')
                    figures.append(fig)

            if show_plots_too:
                plt.show()
    """

