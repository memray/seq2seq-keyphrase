# seq2seq-keyphrase
### Note: this repository has been deprecated. Please move to our latest code/data/model release for keyphrase generation at [https://github.com/memray/OpenNMT-kpg-release](https://github.com/memray/OpenNMT-kpg-release). Thank you.

Data
==========
Check out all datasets at [https://huggingface.co/memray/](https://huggingface.co/memray/).


Introduction
==========
This is an implementation of [Deep Keyphrase Generation](http://memray.me/uploads/acl17-keyphrase-generation.pdf) based on [CopyNet](https://github.com/MultiPath/CopyNet).

One training dataset (**KP20k**), five testing datasets (**KP20k, Inspec, NUS, SemEval, Krapivin**) and one pre-trained model are provided. 

Note that the model is trained on scientific papers (abstract and keyword) in Computer Science domain, so it's expected to work well only for CS papers.


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
