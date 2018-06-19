# Source code for Multi-Modality Disease Modeling via CDMF
In neuroimaging research, brain networks derived from different tractography methods may lead to different results and perform differently when used in classification tasks. As there is no ground truth to determine which brain network models are most accurate or most sensitive to group differences, we developed a new sparse learning method that combines information from multiple network models. We used it to learn a convex combination of brain connectivity matrices from multiple different tractography methods, to optimally distinguish people with early mild cognitive  impairment from healthy control subjects, based on the structural connectivity patterns. 

## Usage
The main entrance of the program is `example_3modalities.py`. This is an example of fusion 3 modalities. In the load function, please load the data files as follows: 

- `x1`: first modality

- `x2`: second modality

- `x3`: third modality

- `y`: labels

- `train_size`: the number of training samples 


## Citation

As you use this code for your exciting discoveries, please cite the paper below:

> Qi Wang, Mengying Sun, Liang Zhan, Paul Thompson, Shuiwang Ji, and Jiayu Zhou. 2017. Multi-Modality Disease Modeling via Collective Deep Matrix Factorization. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '17). 

Or if you use Bibtex:

```
@inproceedings{Wang:2017:MDM:3097983.3098164,
 author = {Wang, Qi and Sun, Mengying and Zhan, Liang and Thompson, Paul and Ji, Shuiwang and Zhou, Jiayu},
 title = {Multi-Modality Disease Modeling via Collective Deep Matrix Factorization},
 booktitle = {Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
 pages = {1155--1164},
 publisher = {ACM},
}
```

The paper can be downloaded [here](http://www.kdd.org/kdd2017/papers/view/multi-modality-disease-modeling-via-collective-deep-matrix-factorization). 
