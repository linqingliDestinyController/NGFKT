# Knowledge Relation Rank Enhanced Heterogeneous Learning Interaction Modeling for Neural Graph Forgetting Knowledge Tracing(NGFKT)

## About 
This is an implementation of the NGFKT model referring to the following paper: Knowledge Relation Rank Enhanced Heterogeneous Learning Interaction Modeling for Neural Graph Forgetting Knowledge Tracing

## Contributors
1. Linqing Li : ll815@uowmail.edu.au
2. Zhifeng Wang : zfwang@ccnu.edu.cn</br>


Faculty of Artificial Intelligence in Education, Central China Normal University, Wuhan 430079, China

## Datasets
We have placed the preprocessed Eedi datasets in the "datasets" folder. When considering the limitations of the github and the large size of the ASSIST2012 datasets, we placed the ASSIST2012 in our google drive referring to the links[<a href="https://drive.google.com/drive/folders/1UO2vVQbrADtX3pybxb4MSMIuGpSEt-A3?usp=share_link">Click</a>]

If you want to process the datasets by yourself, you can reference the corresponding links to download the datasets.</br>
1. ASSIS2012: <a href="https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect">Download</a>
2. Eedi: <a href="https://eedi.com/projects/neurips-education-challenge">Download</a>

## Environment Requirement
python == 3.6.5</br>

tensorflow == 1.15.0</br>

numpy == 1.15.2</br>

## Examples to run the model
When considering the size and format of the datasets, we apply the two file to run the "Eedi" dataset and the "ASSIST2012" dataset respectively.
### Eedi dataset
 Command:</br>
<code data-enlighter-language="raw" class="EnlighterJSRAW">python train_Eedi.py</code>
### ASSIST2012 dataset
 Command:</br>
<code data-enlighter-language="raw" class="EnlighterJSRAW">python train_Assist.py</code>
