from DatasetAndModelUtils import *
import ntpath
import os
TRAIN_MODE = False
EVAL_MODE = False
INFERENCE_MODE = True


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

if __name__ == "__main__":
    downloadedDataset: DownloadedDataset = download_dataset()
    video_classification_model = load_video_classification_model(downloadedDataset=downloadedDataset,
                                                                 load_from_pretrained=not TRAIN_MODE)

    processedDataset: ProcessedDataset = process_datasets(downloadedDataset=downloadedDataset,
                                                          classifModel=video_classification_model)

    if TRAIN_MODE:
        # Sample a train example to see that everything is correct
        processedDataset.test_video_load()

        train_model(video_classification_model,
                    processedDataset,
                    num_epochs=16 if DATASETUSED == DatasetType.DATASET_FIFA else 4)
    elif EVAL_MODE:
        # Inference mode

        # Print debug the model. God bless us
        print(video_classification_model.model)

        # Sample a video from test dataset and check that it is correct
        sample_test_video = next(iter(processedDataset.test))
        processedDataset.investigate_video(sample_test_video)

        # Inference it
        run_inference(videoSamples = [sample_test_video],
                      videoPaths=pathsToEvaluate,
                    modelClassif = video_classification_model,
                    processedDataset = processedDataset,
                      show_output = True)


    elif INFERENCE_MODE:
        from pathlib import PurePath

        def ospath(pathstr: str)-> str:
            res = pathstr.replace("\\", os.sep)
            return res

        if DATASETUSED == DatasetType.DATASET_UCF101:
            pathsToEvaluate = [PurePath(ospath('UCF101_subset\\test\\BasketballDunk\\v_BasketballDunk_g12_c01.avi')),
                                PurePath(ospath('UCF101_subset\\test\\Archery\\v_Archery_g16_c01.avi'))]
        elif DATASETUSED == DatasetType.DATASET_FIFA:
            pathsToEvaluate = [PurePath(ospath('FIFAdataset\\test\\crowd_pos\\2024-01-31 15-41-24.mp4')),
                                PurePath(ospath('FIFAdataset\\test\\crowd_neg\\2024-02-01 10-42-19.mp4')),
                               PurePath(ospath('FIFAdataset\\test\\pitch_neg\\2024-02-01 11-09-13.mp4')),
                               PurePath(ospath('FIFAdataset\\test\\pitch_neg\\2024-02-01 11-09-35.mp4')),
                               PurePath(ospath('FIFAdataset\\test\\pitch_neg\\2024-02-01 11-09-57.mp4')),
                               PurePath(ospath('FIFAdataset\\test\\pitch_neg\\2024-02-01 11-10-41.mp4')),
                               PurePath(ospath('FIFAdataset\\test\\pitch_neg\\2024-02-01 11-12-21.mp4')),
                               PurePath(ospath('FIFAdataset\\test\\pitch_pos\\2024-01-31 15-28-31.mp4')),
                               ]

        videosInModelFormat = [get_video_in_model_format(path, processedDataset) for path in pathsToEvaluate]

        # Inference it
        run_inference(videoSamples=videosInModelFormat,
                      videoPaths=pathsToEvaluate,
                      modelClassif=video_classification_model,
                      processedDataset=processedDataset,
                      show_output=True)
    else:
        assert False, "Unknown Inference Mode"
