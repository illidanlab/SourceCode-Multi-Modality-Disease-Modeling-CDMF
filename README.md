# Source code for Multi-Modality Disease Modeling via CDMF
Alzheimer's disease (AD), one of the most common causes of dementia, is a severe irreversible neurodegenerative disease that results in loss of mental functions. The transitional stage between the expected cognitive decline of normal aging and AD, mild cognitive impairment (MCI), has been widely regarded as a suitable time for possible therapeutic intervention. The challenging task of MCI detection is therefore of great clinical importance, where the key is to effectively fuse predictive information from multiple heterogeneous data sources collected from the patients. In this work, we propose a framework to fuse multiple data modalities for predictive modeling using deep matrix factorization, which explores the non-linear interactions among the modalities and exploits such interactions to transfer knowledge and enable high performance prediction. Specifically, the proposed collective deep matrix factorization decomposes all modalities simultaneously to capture non-linear structures of the modalities in a supervised manner, and learns a modality specific component for each modality and a modality invariant component across all modalities. The modality invariant component serves as a compact feature representation of patients that has high predictive power. The modality specific components provide an effective means to explore imaging genetics, yielding insights into how imaging and genotype interact with each other non-linearly in the AD pathology. Extensive empirical studies using various data modalities provided by Alzheimer's Disease Neuroimaging Initiative (ADNI) demonstrate the effectiveness of the proposed method for fusing heterogeneous modalities.  

## Usage
`example_3modalities.py` is an example of fusion 3 modalities. In the function 'load_data', please load the data files as follows: 

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
