# Source code for TREX
This paper proposes TREX framework to reveal the short-cut to beyond-accuracy metrics (fairness, diversity) in next basket recommendation (NBR).

This repository is built based on the following repositories:

Measuring Fairness in Ranked Results: An Analytical and Empirical. https://github.com/BoiseState/rank-fairness-metrics

A Next Basket Recommendation Reality Check. https://github.com/liming-7/A-Next-Basket-Recommendation-Reality-Check

Understanding Diversity in Session-based Recommendation. https://github.com/qyin863/Understanding-Diversity-in-SBRSs



## Required packages
To run our data preprocessing and evaluation scripts, Pandas, Numpy and Python >= 3.6 are required.

To run the pubished NBR methods' code, please go to the original repository and check the required packages.



## Code structure
* preprocess: contains the script of dataset preprocessing and splitting. 
* csvdata: contains the .csv format dataset after preprocessing.
* jsondata: contains the .json format dataset after preprocessing, history baskets sequence and future basket are stored seperately.
* mergedata: contains the .json format dataset after preprocessing, history baskets sequence and future basket are stored together.
* methods: contains the source code of different NBR methods and the original url repository link of these methods.
* evaluation: scripts for evaluation.
    * fair_metrics: the fairness metrics.
    * metrics.py: the accuracy metrics.
    * model_performance.py: evaluate the fairness and accuracy of recommendation results.


## Pipeline
* Step 1. Preprocess and split the datasets. Generate different types of preprocessed datasets for different NBR methods.
* Step 2. Train the model and save the model. (Note that we use the original implementations of the authors, so we provide the original repository links, which contain the instructions of the environment setting, how to run each method, etc. We also provide our additional instructions and hyperparameters in the following section, which can make the running easier.)
* Step 3. Generate the predicted results via the trained model and save the results file.
* Step 4. Use the evaluation scripts to get the performance results.

## Dataset 
### Preprocessing
We provide the scripts of preprocessing, and the preprocessed dataset with different formats (csvdata, jsondata, mergedata), which can be used directly.
If you want to preprocess the dataset yourself, you can download the dataset from the following urls and put them into the "rawdata/{dataset}" folder.
* Instacart: https://www.kaggle.com/c/instacart-market-basket-analysis/data
* Dunnhumby: https://www.dunnhumby.com/source-files/



### Format description of preprocessed dataset
* csvdata: --> G-TopFreq, P-TopFreq, GP-TopFreq, ReCANet
> user_id, order_number, item_id, basket_id

* jsondata: --> TIFUKNN, DNNTSP, DREAM

> history data: {uid1: [[-1], basket, basket, ..., [-1]], uid2:[[-1], basket, basket, ..., [-1]], ... }

> future data: {uid1: [[-1], basket, [-1]], uid2: [[-1], basket, [-1]], ...}

* mergedata: --> UP-CF

> {uid1: [basket, basket, ..., basket], uid2: [basket, basket, ..., basket], ...}

### Format description of predicted results
* Predicted items:

> {uid1: [item, item, ..., item], uid2: [item, item, ..., item], ...}

* Predicted relevance scores (for all the items):

> {uid1: [rel, rel, ..., rel], uid2: [rel, rel, ..., rel], ...}





## TREX

### Step1: get repeat results
```
python repetition/repeat.py --dataset instacart --alpha 0.3 --beta 0.8
python repetition/repeat.py --dataset dunnhumby --alpha 0.7 --beta 0.9

```
### Step2: get explore results

```
python exploration/ex-fair.py --dataset instacart

```
### Step3: generate the final basket
```
python basket_generation.py --dataset instacart
python basket_generation_div.py --dataset instacart

```
### Step4: evaluate
```
python evaluation/model_performance.py --pred_folder XXX --number_list 0 --method trex-fair --threshold 0.0

```












