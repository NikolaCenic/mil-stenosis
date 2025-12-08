# Multi-View Stenosis Classification Leveraging Transformer-Based Multiple-Instance Learning
<p align="center">
  <img src="main_figure.png" alt="Figure" width="50%"/>
</p>
# Abstract
Coronary artery stenosis is a leading cause of cardiovascular disease, diagnosed by analyzing the coronary arteries from multiple angiography views. Although numerous deep-learning models have been proposed for stenosis detection from a single angiography view, their performance heavily relies on expensive view-level annotations, which are often not readily available in hospital systems. Moreover, these models fail to capture the temporal dynamics and dependencies among multiple views, which are essential aspects 
for clinical diagnosis. To address this, we propose SegmentMIL, a transformer-based multi-view multiple-instance learning framework for patient-level stenosis classification. Trained on a real-world clinical dataset, using patient-level supervision and without any view-level annotations, SegmentMIL jointly predicts the presence of stenosis and localizes the affected anatomical region, distinguishing between the right and left coronary arteries and their respective segments. SegmentMIL obtains high performance on internal and external evaluations and outperforms both view-level models and classical MIL baselines, underscoring its potential as a clinically viable and scalable solution for coronary stenosis diagnosis. 
# Create Environment
The repo requirments are stored in `environment.yml`. 
To recreate the conda environment run: `source setup.sh`
# Training
The training is configured in `configs/config.yaml`. The hyperparameters in the config are set according to our experimental results andadapted for running on a single NVIDIA A40.

To run the default configuration of SegmentMIL just run: 
`python train.py`

To run with different configuration run:
`python train.py --parameter=value`


The trained models get stored in `stenosis-classification\{timestamp}-{run_id}`.
# Evaluation

For evaluation of a trained model, run the following command:
`python evaluate.py --model-dir trained_model_dir`. 

By default the evaluation is done at the `last.pt` checkpoint from the `trained_model_dir` on the patient level on the internal test dataset. To evaluate on the CADICA dataset add `--dataset test_cadica`. To do view level evaluation add `--evaluation-level ViewLevel`.

