# Fake News Detection
> Upenn 2023Fall CIS 5300 final project

Fake news detection algorithms play a pivotal role in safeguarding the integrity of information in our digital age. With the exponential growth of information on social media platforms, the ability to discern between authentic and deceptive content is crucial. In this study, binary classifiers for fake news detection are trained on the LIAR and FNC-1 datasets, and their performance against published baseline works is evaluated. The model with the best results achieved an accuracy of #TODO and F1 score of #TODO, by utilizing an ensemble model of BERT and LSTM. 


## Authors(ordered by alphabet): 
- [Kaiwen Hu](https://github.com/kevin00hu)
- [Yijia Xue](https://github.com/Artyxi)
- [Yuzhuo Kang](https://github.com/andykang8099)
- [Zhixuan Li](https://github.com/zhxabi)

## structure

### milestone 1
It contains the [data](https://github.com/kevin00hu/CIS-5300-final-project/blob/master/milestone%201/dataset/data.md) we used for this project: 
- [FNC](http://www.fakenewschallenge.org/) 
- [LIAR](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)

### mile1stone 2
It contains two baseline models: 
- A 4-layers Feed Forward Neural Network with only LIAR features: `baseline.py`
- A 4-layers Feed Forward Neural Network with LIAR features + feature extract from LIAR using model trained with FNC data: `baseline_FNC_model.ipynb`
- [Result](https://github.com/kevin00hu/CIS-5300-final-project/blob/master/milestone%202/baseline/score.md) for those models.
  
### milestone 3
It contains two BERT models:
- basic bert model: `bert_base.py`
- fine-tuned bert model: `extension-1.py`
- [Result](https://github.com/kevin00hu/CIS-5300-final-project/blob/master/milestone%203/code/score.md) for thsoe models.

### milestone 4
It contains a BERT + LSTM model:
