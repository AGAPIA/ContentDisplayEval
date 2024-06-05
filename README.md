# ContentDisplayEval
A framework for testing the correctness of the visualizations rendered by in-game cameras


##  Part 1: Folder VideoClassification contains the video content evaluation methods. 
The trained model can be found at https://tinyurl.com/ModelVideoClassification.
A sample dataset can be found at https://tinyurl.com/DatasetVideo.
Just download the content inside the VideoClassification folder and give it a go. 
The main script is VideoClassification.py, take a look at the args to see the various options you have for training/inference first. It needs to point to the local working folder VideoClassification to have paths working correctly. 
If you use PyCharm, there are already two configurations out-of-the-box for this, one for train and one for evaluation.
The evaluation will output a csv file containing the results for each of the "val" item in the dataset sample. The output can be found in results/ subfolder.


## Part 2: ImageBasedEvaluation model.

The model itself needs to be downloaded from https://tinyurl.com/ImageBasedModel, while the sampled dataset from https://tinyurl.com/fhpaax69. As above, copy the content in the subfolder.
There are two notebookds to handle model training/inference inside the folder.

## Part 3: TODO:

We plan to release an open source full pipeline to utilize the two models in a public game engine such as Unreal Engine. Currently the pipeline stays in proprietary code. However, the techniques described in the paper works independently of any game engine foundation.


