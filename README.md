# seq2seq-keyphrase


Introduction
==========
This is an implementation of [Deep Keyphrase Generation](http://memray.me/uploads/acl17-keyphrase-generation.pdf) based on [CopyNet](https://github.com/MultiPath/CopyNet).

One training dataset (**KP20k**), five testing datasets (**KP20k, Inspec, NUS, SemEval, Krapivin**) and one pre-trained model are provided. 

Note that the model is trained on scientific papers (abstract and keyword) in Computer Science domain, so it's expected to work well only for CS papers.

Run
==========

### Entry and Settings
The main entry of the project is placed in `keyphrase/keyphrase_copynet.py`

All the primary settings are stored in `keyphrase/config.py`. Training and Prediction load settings from [`setup_keyphrase_all()`](https://github.com/memray/seq2seq-keyphrase/blob/master/keyphrase/config.py#L7) and Evaluation loads the setting from [`setup_keyphrase_baseline()`](https://github.com/memray/seq2seq-keyphrase/blob/070ff8fc4abb51b96e4935934e70b8d92c6666f6/keyphrase/config.py#L191).

Some scripts for data processing are placed in `keyphrase/dataset/`. 

### Before running...
Before running the code, please download this [experiment_dataset.zip](https://drive.google.com/file/d/0B-7HD48qQzxVQkl5UHBZZVJQM00/view?usp=sharing), unzip it to the project directory and overwrite the `Experiment/` and `dataset/`. 

`Experiment/` contains one pre-trained copy-seq2seq model *(experiments.keyphrase-all.one2one.copy.id=20170106-025508.epoch=4.batch=1000.pkl)* used in the paper, based on which you can extract keyphrases for your own corpus.
Besides there are some output examples in this folder. 

`dataset/` contains three folders.
   * `punctuation-20000validation-20000testing` contains the pickled data for training/validation.
   * `testing-data` contains the original testing datasets, and they are further processed into folder `baseline-data`.
   * `baseline-data` stores the cleaned and processed testing datasets, and will be used during predicting and evaluating. Specifically, for each dataset, there's one `text` folder contains the content of paper after POS-tagging, and another `keyphrase` folder contains the ground-truth keyphrases, listed one phrase per line.

### Training
If you want to train a new model, set *config['do_train'] = True* and *config['trained_model'] = ''*  in `keyphrase/config.py`. 

If the *config['trained_model']* is not empty, it will load the trained model first and resume training. 

 Also, there are some parameters you can try out, like *config['copynet'] = False* means to train a normal GRU-based Seq2seq model.

### Predicting keyphrases
Set *config['do_predict'] = True* and *config['testing_datasets']=['data_set1', 'data_set2' ...]* (datasets you wanna extract). The program will load the text from `dataset/baseline-data/` first, and save the prediction results into `config['predict_path']/predict.generative.dataset_name.pkl` and the extracted phrases into `dataset/keyphrase/prediction/`.

Similarly, there are many parameters to tune the prediction of keyphrase.

If you want to extract keyphrases from your own data using our model, you need to put your data in `baseline-data` following the same format, and implement a simple class in `keyphrase/dataset/keyphrase_test_dataset.py`.

### Test
Set *config['do_evaluate'] = True* and you'll see a lot of print-outs in the console and reports in directory `config['predict_path']`. Please be aware that this result is only for developing and debugging and it's slightly different from the reported result.

### Evaluation (to reproduce the results in the paper)
The performances reported in the paper is done by `keyphrase/baseline/evaluate.py`. It loads the phrases from `dataset/keyphrase/prediction/` and evalutes them with metrics such as Precision, Recall, F-score, Bpref, MRR etc.

Note that the setting of evaluation is different from the settings used in training/predicting and don't be confused. It is loaded by calling `setup_keyphrase_baseline()` in `config.py`. Also if you want to reproduce the result of present keyphrase prediction (Section 5.1 of the paper), please set `config['predict_filter']` and `config['target_filter']` to 'appear-only' (line 292,293). Similarly, set them to 'non-appear-only' for reproducing absent keyphrase prediction (Section 5.2 of the paper). 

You can find the awesome baseline implementations from [Kazi Saidul Hasan](http://www.hlt.utdallas.edu/~saidul/code.html) (TfIdf, TextRank, SimpleRank, ExpandRank) and [Alyona Medelyan](http://www.medelyan.com/software) (Maui and KEA). I also have put my keyphrase outputs [here](https://drive.google.com/open?id=1DsAno3lvMlr-5rCk4qohcy4U3m__Sfj8) for your convenience (unzip to `seq2seq-keyphrase-release/dataset/keyphrase/prediction`). 

Data
==========
The training data mentioned above is pickled. You can download here: [experiment_dataset.zip](https://drive.google.com/file/d/0B-7HD48qQzxVQkl5UHBZZVJQM00/view?usp=sharing). Just in case you are in China Mainland where downloading this large file is painful, I provide another [link](http://pan.baidu.com/s/1b46mwY) on Baidu Pan Cloud Drive. 

If you are just interested in using the KP20k dataset, you can get the data as well: [kp20k.zip](https://drive.google.com/open?id=0B-7HD48qQzxVd2plbUlGdFduRDQ). 

The KP20k dataset is released in JSON format. Each data point contains the title, abstract and keywords of a paper.

Part | #(data) 
--- | --- 
Training | 530,803 
Validation | 20,000
Test | 20,000

The raw dataset (without filtering noisy data) is also provided. Please download [here](https://drive.google.com/open?id=0B-7HD48qQzxVQ3hIMW5NY1RWQ0E). 

Cite
==========
If you use the code or datasets, please cite the following paper:

> Rui Meng, Sanqiang Zhao, Shuguang Han, Daqing He, Peter Brusilovsky and Yu Chi. Deep Keyphrase Generation. 55th Annual Meeting of Association for Computational Linguistics, 2017. [[PDF]](http://memray.me/uploads/acl17-keyphrase-generation.pdf) [[arXiv]](https://arxiv.org/abs/1704.06879)

```
@InProceedings{meng-EtAl:2017:Long,
  author    = {Meng, Rui  and  Zhao, Sanqiang  and  Han, Shuguang  and  He, Daqing  and  Brusilovsky, Peter  and  Chi, Yu},
  title     = {Deep Keyphrase Generation},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month     = {July},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
  pages     = {582--592},
  url       = {http://aclweb.org/anthology/P17-1054}
}
```