

### baseline structure
- dataset.py: This file does the preprocessing on two dataset mentioned in `milestone1`.
- score.py: Contains the evaluation metrics and confusion matrix plot used by baselines.
- bert_base.py:   Evaluating testset on pretrained BERT model - *bert-base-uncased*
- extension-1.py: Training script for BERT model.


### Running instruction
Prerequire python package:

`pip install -r requirements.txt`

First we need to get into the code directory:

`cd "milestone 3/code"`

bert_base.py:

`python3 bert_base.py`

extension-1.py:

`python3 extension-1.py`

score.py:

`python3 score.py ./output/y_test_baseline.npy ./output/y_pred_baseline.npy` 


### Result
The confusion matrix and loss trending plots are saved in `./plots` directory.
| Model                  | Accuracy | F1 Score |
|------------------------|----------|----------|
| Baseline               | 64.72%   | 0.40     |
| FNC_model              | 64.85%   | 0.51     |
| Baseline + FNC_feature | 62.11%   | 0.45     |
| LSTM                   | 61.64%   | 0.52     |
| Bert                   | 68.63%   | 0.56     |
