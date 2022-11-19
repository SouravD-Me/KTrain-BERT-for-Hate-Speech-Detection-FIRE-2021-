# Probabilistic Impact Score Generation using Ktrain-BERT to Identify Hate Words from Twitter Discussions


This repository contains the source code for experimentation and the public dataset (train and test) of the Hate Speech and Offensive Content Identification in English and Indo-Aryan Languages from 13th meeting of Forum for Information Retrieval Evaluation (FIRE 2021).

The paper for the same is:

## [**Das, S., Mandal, P., & Chatterji, S. (2021). Probabilistic Impact Score Generation using Ktrain-BERT to Identify Hate Words from Twitter Discussions. arXiv preprint arXiv:2111.12939.**](https://arxiv.org/ftp/arxiv/papers/2111/2111.12939.pdf)

Or,

## [FIRE 2021 Working Notes](https://ceur-ws.org/Vol-3159/)

***


## Abstract: 
<div align="justify">
Social media has seen a worrying rise in hate speech in recent times. Branching to several distinct categories of cyberbullying, gender discrimination, or racism, the combined label for such derogatory content can be classified as toxic content in general. This paper presents experimentation with a Keras wrapped lightweight BERT model to successfully identify hate speech and predict probabilistic impact score for the same to extract the hateful words within sentences. The dataset used for this task is the Hate Speech and Offensive Content Detection (HASOC 2021) data from FIRE 2021 in English. Our system obtained a validation accuracy of 82.60%, with a maximum F1-Score of 82.68%. Subsequently, our predictive cases performed significantly well in generating impact scores for successful identification of the hate tweets as well as the hateful words from tweet pools.



***

## Brief Methodology: 

For the proposed research, we choose only the English dataset from [HASOC Subtask 1](https://hasocfire.github.io/hasoc/2022/dataset.html). In order to shape the data for our proposed work, we introduce a set of extensive data analysis steps and reduce the data attributes only to the atomic columns. We split the train and test sets using linear Logistic Regression, and save the model. Next, we incorporate the ktrain BERT model **[1]**, which is a lightweight BERT wrapped by the Tensorflow-Keras2 library for low resource systems and a faster training phase. We extend the model further for several hyperparameters tuning, train and test visualization, and most importantly for probabilistic impact score generation from the trained model for any random sentences, to point out the hate speech factor (word) in those sentences. Finally, we produce the qualitative analysis by classification report from the saved model. 

## Workflow:

<div align="center">

  ![Model Diagram](https://user-images.githubusercontent.com/63003115/202840186-d833e803-8d20-4b88-a19a-c9a86091d0ec.png)

</div>


***

## Probabilistic Impact Score Generation:

After the validation phase, we choose random tweets both from the training and test data to explain the model on a predictive probabilistic scale. It is obtained by extending the trained set and validating the test data further by fusing the explain and predict function on the randomly fed tweets. We select three tweets with hate content, and the other three with non-hateful content (see the paper). The scores are normalized in a comparable scale, i.e., all the scores are in a non-negative scale, while quite naturally the higher scores represent more positive or negative impacts, reliant on the context of the concerning tweet itself.

An example from both hateful and non-hateful tweets are:

### Non-Hateful Tweet:

<div align="center">

 ![POS3](https://user-images.githubusercontent.com/63003115/202840493-d531aac1-0601-4165-be34-52cd5086b0b3.png)
</div>


### Hateful Tweet:

<div align="center">

 ![NEG2](https://user-images.githubusercontent.com/63003115/202840524-a6c314c0-4e81-47ee-ab02-6bdd08bb478d.png)

</div>

***

## Detailed Classification Report:

| <div align="center"> Measure </div> | <div align="center"> Value </div> | <div align="center"> Derivations </div> |
| ------------- | ------------- | ------------ |
| Sensitivity |	0.8037 |	TPR = TP / (TP + FN) |
| Specificity |	0.8535 |	SPC = TN / (FP + TN) |
| Precision   | 0.8716 |	PPV = TP / (TP + FP) |
| Negative Predictive Value |	0.7785 |	NPV = TN / (TN + FN) |
| False Positive Rate |	0.1465 | FPR = FP / (FP + TN) |
| False Discovery Rate | 0.1284 | FDR = FP / (FP + TP) |
| False Negative Rate | 0.1963 |	FNR = FN / (FN + TP) |
| Accuracy |	0.8260 |	ACC = (TP + TN) / (P + N) |
| F1 Score |	0.8363 |	F1 = 2TP / (2TP + FP + FN) |
|Matthews Correlation Coefficient |	0.6536 |	TP*TN - FP * FN / sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN)) |


***

## Please cite the following paper if you utilize this dataset and/or the source-code in your research:

@article{das2021probabilistic,
  title={Probabilistic Impact Score Generation using Ktrain-BERT to Identify Hate Words from Twitter Discussions},
  author={Das, Sourav and Mandal, Prasanta and Chatterji, Sanjay},
  journal={arXiv preprint arXiv:2111.12939},
  year={2021}
}


***

## Declaration:

The hate speeches shown for the probabilistic impact score demonstrations are for experimental purposes only. These are real tweets and are collected from the training and/or test data. The authors DO NOT promote any form of hate content on social networks, and strongly condemn it.

***

[1] [ktrain: A Low-Code Library for Augmented Machine Learning](https://arxiv.org/abs/2004.10703)
