# source code for trex-framework

#### Step1: generate k fold files
```
python keyset_fold.py
```
#### Step2: train exploration module and pred exploration results (exploration folder)
```
python train_main.py
python pred_results.py
```
#### Step3: get repetition results (repetition folder)

```
python hyper_modified.py
```
#### Step4: generate the final basket (rfgr folder)
```
python basket_generation.py
```

Need to specify the files inside the code.
