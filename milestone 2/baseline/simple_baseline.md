### Prerequest

Please down load the [GoogleNews-vectors-negative300.bin.gz](https://github.com/mmihaltz/word2vec-GoogleNews-vectors) into this directory.

### baseline structure
- dataset.py: This file does the preprocessing on two dataset mentioned in `milestone1`.
- baseline.py: This file define a four layers feed forward neural network, and training using word2vec data from LIAR only.
- baseline_FNC_model.ipynb: Training another model using FNC model based on joined news title and news body after word2vec average embedding.
- baseline_FNC_feature.ipynb: Trainging a third model where the dataflow as below:
  - LIAR average word embedding(None, 300) -> FNC_model(None, 2) +  LIAR(None, 300) -> predict result(None, 2)
  - each `->` means a model, `+` means the data concatenation process(torch.concat with axis = 1)
  
### Result
The confusion matrix and loss trending plots are saved in `./plots` directory.
| model     | accuracy | f1 score|
| --------- | -------- | ------- |
| baseline  | 64.72%   | 0.40    |
| FNC_model | 64.85%   | 0.51    |
| base_line + FNC_feature | 62.11% | 0.45 |