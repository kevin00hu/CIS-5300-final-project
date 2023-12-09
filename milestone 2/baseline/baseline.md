

### baseline structure
- dataset.py: This file does the preprocessing on two dataset mentioned in `milestone1`.
- baseline.py: This file define a four layers feed forward neural network, and training using word2vec data from LIAR only.
- baseline_FNC_model.ipynb: Training another model using FNC model based on joined news title and news body after word2vec average embedding.
- baseline_FNC_feature.ipynb: Trainging a third model where the dataflow as below:
  - LIAR average word embedding(None, 300) -> FNC_model(None, 2) +  LIAR(None, 300) -> predict result(None, 2)
  - each `->` means a model, `+` means the data concatenation process(torch.concat with axis = 1)
- stong_baseline.py: Training a LSTM model as our strong baseline. 
- score.py: Contains the evaluation metrics and confusion matrix plot used by baselines
### Running instruction
Prerequire python package:

`pip install -r requirements.txt`

First we need to get into the the code directory:

`cd "milestone 2/baseline"`

baseline.py:

`python3 baseline.py`

baseline_FNC_model.ipynb, baseline_FNC_feature_.ipynb:

These two files are jupyter notebook, which can simply use `run all`.

stong_baseline.py:

`python3 strong_baseline.py` 

score.py:

`python3 score.py ./output/y_test_baseline.npy ./output/y_pred_baseline.npy` 


### Result
The confusion matrix and loss trending plots are saved in `./plots` directory.
| model     | accuracy | f1 score|
| --------- | -------- | ------- |
| baseline  | 64.72%   | 0.40    |
| FNC_model | 64.85%   | 0.51    |
| base_line + FNC_feature | 62.11% | 0.45 |
| LSTM | 64.56% | 0.35 |