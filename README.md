# Source code for TREx
This paper proposes TREx framework to reveal the short-cut to beyond-accuracy metrics (fairness, diversity) in next basket recommendation (NBR).

We additionally:
* reproduce 8 NBR methods 
* evaluate the performance using 5 fairness metrics, 3 diversity metrics and 3 accuracy metrics.


## Required packages
To run our TREx scripts, Pandas, Numpy and Python >= 3.10 are required.

To run the published NBR methods' code, please go to the original repository and check the required packages.

## Dataset 

* Instacart: https://www.kaggle.com/c/instacart-market-basket-analysis/data
* Dunnhumby: https://www.dunnhumby.com/source-files/

We provide the preprocessed dataset with different formats (csvdata, jsondata, mergedata), which can be used directly.

### Format description of preprocessed dataset
* csvdata: --> G-TopFreq, P-TopFreq, GP-TopFreq, ReCANet, TREx
> user_id, order_number, item_id, basket_id

* jsondata: --> TIFUKNN, DNNTSP, DREAM, TREx

> history data: {uid1: [[-1], basket, basket, ..., [-1]], uid2:[[-1], basket, basket, ..., [-1]], ... }

> future data: {uid1: [[-1], basket, [-1]], uid2: [[-1], basket, [-1]], ...}

* mergedata: --> UP-CF

> {uid1: [basket, basket, ..., basket], uid2: [basket, basket, ..., basket], ...}

### Format description of predicted results
* Predicted items:

> {uid1: [item, item, ..., item], uid2: [item, item, ..., item], ...}



## Code structure

* csvdata, jsondata, mergedata: contain different dataset formats.
* repetition: contains scripts of repetition module.
* exploration: contains scripts of explore-fair.
* combine: contains scripts for generating the final basket (trex-fair, trex-div).
* evaluation: scripts for evaluation.
    * fair_metrics: the fairness metrics.
    * diversity_metrics.py: the diversity metrics. 
    * metrics.py: the accuracy metrics.
    * model_performance.py: evaluate the fairness, diversity and accuracy of recommendation results.
* methods: contains 8 NBR methods.

## TREX

Step1: get repeat results, which is saved as 'repeat_result/{dataset}_pred.json'
```
python repetition/repeat.py --dataset instacart --alpha 0.3 --beta 0.8
python repetition/repeat.py --dataset dunnhumby --alpha 0.7 --beta 0.9

```


Step2: get ex-fair results, which is saved as 'ex-fair_result/{dataset}_pred.json'

```
python exploration/ex-fair.py --dataset instacart

```


Step3: generate the final basket

For TREx-Fair, the results are saved as 'final_results_fair/{dataset}_ pred_{threshold}.json'
```
python basket_generation.py --dataset instacart

```

For TREx-Div, the following command generates ex-div results and the final basket:
```
python basket_generation_div.py --dataset instacart

```
The results are saved as 'final_results_div/{dataset}_ pred_{threshold}.json'


Step4: evaluate
```
python evaluation/model_performance.py --pred_folder XXX --number_list 0 --method XXX --threshold XX

```

XXX is the folder where you put the predicted results. The evaluation results are saved as 'eval_trex-fair/eval_{method_name}_{threshold}.txt'


## Guidelines for NBR baselines
Our reproducibility relies as much as possible on the artifacts provided by the user themselves, the following repositories have information about how to run each NBR method and the required packages.
* UP-CF@r: https://github.com/MayloIFERR/RACF
* TIFUKNN: https://github.com/HaojiHu/TIFUKNN
* DREAM: https://github.com/yihong-chen/DREAM
* DNNTSP: https://github.com/yule-BUAA/DNNTSP
* ReCANet: https://github.com/mzhariann/recanet

We also provide our additional instructions if the original repository is not clear, as well as the hyperparameters we use.

We set five random seed: 12345, 12321, 54321, 66688, 56789. And the corresponding number of the predicted files are 0, 1, 2, 3, 4.
For G-TopFreq, P-TopFreq, GP-TopFreq, TIFUKNN, the predicted results of each run are same and not influenced by the random seed. Therefore, we only keep one set of predicted files with number 0.

Please create a folder "results" under each method to store the predicted files.

### G-TopFreq, P-TopFreq, GP-TopFreq
Three frequency based methods are under the folder "methods/g-p-gp-topfreq".
* Step 1: Check the file path of the dataset.
* Step 2: Using the following commands to run each method:
```
python g_topfreq.py --dataset instacart 
...
python p_topfreq.py --dataset instacart
...
python gp_topfreq.py --dataset instacart
...
```
Predicted files are stored under folder: "g_top_results", "p_top_results", "gp_top_results".

Predicted file name: {dataset}_pred0.json, {dataset}_rel0.json

### UP-CF@r
UP-CF@r is under the folder "methods/upcf".
* Step 1: Check the dataset path and keyset path.
* Step 2: Predict and save the results using the following commands:
```
python racf.py --dataset instacart --recency 5 --asymmetry 0.25 --locality 5 --seed 12345 --number 0
...
python racf.py --dataset dunnhumby --recency 25 --asymmetry 0.25 --locality 5 --seed 12345 --number 0
...

``` 
Predicted file name: {dataset}_pred{number}.json, {dataset}_rel{number}.json

### TIFUKNN
TIFUKNN is under the folder "methods/tifuknn"
* Step 1: Predict and save the results using the following commands:
```
python tifuknn_new.py ../jsondata/instacart_history.json ../jsondata/instacart_future.json ../keyset/instacart_keyset.json 900 0.9 0.6 0.7 3 20 
...
python tifuknn_new.py ../jsondata/dunnhumby_history.json ../jsondata/dunnhumby_future.json ../keyset/dunnhumby_keyset.json 100 0.9 0.9 0.1 7 20 
...

```
Predicted file name: {dataset}_pred0.json, {dataset}_rel0.json

### Dream
Dream is under the folder "methods/dream".
* Step 1: Check the file path of the dataset in the config-param file "{dataset}conf.json"
* Step 2: Train and save the model using the following commands:
```
python trainer.py --dataset instacart --attention 1 --seed 12345 
...
python trainer.py --dataset dunnhumby --attention 1 --seed 12345 
...

```
* Step 3: Predict and save the results using the following commands:
```
python pred_results.py --dataset instacart --attention 1 --seed 12345 --number 0
...
python pred_results.py --dataset dunnhumby --attention 1 --seed 12345 --number 0
...

```
Predicted file name: {dataset}_pred{number}.json, {dataset}_rel{number}.json


### DNNTSP
DNNTSP is under the folder "methods/dnntsp".
* Step 1: Confirm the name of config-param file "{dataset}config.json" in ../utils/load_config.py. Check the file path of the dataset in the corresponding file "../utils/{dataset}conf.json". For example:
```
abs_path = os.path.join(os.path.dirname(__file__), "instacartconfig.json")
with open(abs_path) as file:
    config = json.load(file)
```
```
{
    "data": "Instacart",
    "save_model_folder": "DNNTSP",
    "history_path": "../jsondata/instacart_history.json",
    "future_path": "../jsondata/instacart_future.json",
    "keyset_path": "../keyset/instacart_keyset_0.json",
    "items_total": 29399,
    "item_embed_dim": 16,
    "cuda": 0,
    "loss_function": "multi_label_soft_loss",
    "epochs": 40,
    "batch_size": 64,
    "learning_rate": 0.001,
    "optim": "Adam",
    "weight_decay": 0
}
```
* Step 2: Train and save the models using the following command:
```
python train_main.py --seed 12345
```
* Step 3: Predict and save results using the following commands:
```
python pred_results.py --dataset instacart --number 0 --best_model_path XXX
```
Note, DNNTSP will save several models during the training, an epoch model will be saved if it has higher performance than previous epoch, so XXX is the path of the last model saved during the training.

Predicted file name: {dataset}_pred{number}.json, {dataset}_rel{number}.json



### ReCANet
ReCANet is under the folder "methods/recanet"
* Step 1: Predict and save the results using the following commands:
```
python main.py -dataset instacart -user_embed_size 64 -item_embed_size 16 -hidden_size 64 -history_len 35 -number 0 -seed_value 12345
...
python main.py -dataset dunnhumby -user_embed_size 16 -item_embed_size 128 -hidden_size 64 -history_len 35 -number 0 -seed_value 12345 
...

```
Predicted file name: {dataset}_pred{number}.json, {dataset}_rel{number}.json







