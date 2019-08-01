## Introduction



Implementation for ICLR2019 paper **MAX-MIG: AN INFORMATION THEORETIC APPROACH FOR JOINT LEARNING FROM CROWDS**

paper link: https://openreview.net/forum?id=BJg9DoR9t7

arxiv : https://arxiv.org/abs/1905.13436





## Synthesized Crowd-sourcing dataset



- To run experiments of *Dogs vs. Cats* dataset in `Dogs vs. Cats` directory: 

```shell
python3 main.py --case case_num --expertise ex --path path_to_dataset --device device_num

case_num: number of experimental case( see our paper) 
		1: Independent mistakes
		2: Naive majority
		3: Correlated mistakes
expertise: the expertise of senior expertise
		0: Low expertise
		1: High expertise
		
path_to_dataset: path to the dataset

device_num : GPU number
		
```



- To run experiments of *CIFAR-10* dataset in `Cifar10` directory: 

```shell
python3 main.py --case case_num --expertise ex --path path_to_dataset --device device_num

case_num: number of experimental case( see our paper) 
		1: Independent mistakes
		2: Naive majority
		3: Correlated mistakes
expertise: the expertise of senior expertise
		0: Low expertise
		1: High expertise
		
path_to_dataset: path to the dataset

device_num : GPU number
		
```





- To run experiments of *LUNA* dataset in `LUNA16` directory: 

```shell
python3 main.py --case case_num --expertise ex --path path_to_dataset --device device_num

case_num: number of experimental case( see our paper) 
		1: Independent mistakes
		2: Naive majority
		3: Correlated mistakes
expertise: the expertise of senior expertise
		0: Low expertise
		1: High expertise
		
path_to_dataset: path to the dataset

device_num : GPU number
		
```





## Real world crowd-sourcing dataset:



- To run experiments of *Labelme* dataset in `labelme` directory: 

```shell
python3 cotraining_labelme.py  --device device_num

device_num : GPU number
		
```



The Labelme dataset can be downloaded at http://fprodrigues.com//deep_LabelMe.tar.gz . Please place `prepared` document in the same folder with your code. :smile:



To cite our paper:

```shell
@article{cao2018max,

  title={Max-MIG: an Information Theoretic Approach for Joint Learning from Crowds},

  author={Cao, Peng and Xu, Yilun and Kong, Yuqing and Wang, Yizhou},

  year={2018}

}

```







