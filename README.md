## Shedding Light on Problems with Hyperbolic Graph Learning

We provide the code for [Shedding Light on Problems with Hyperbolic Graph Learning](https://arxiv.org/pdf/2411.06688) in this repository.

Summary: Our work makes a surprising discovery: when simple Euclidean models with comparable numbers of parameters are properly trained in the same environment, in most cases, they perform as well, if not better, than all introduced hyperbolic graph representation learning models, even on graph datasets previously claimed to be the most hyperbolic as measured by Gromov $\delta$-hyperbolicity (i.e., perfect trees). This observation gives rise to a simple question: how can this be? We answer this question by taking a careful look at the field of hyperbolic graph representation learning as it stands today, highlight a number of crucial issues and resolve a core subset of them in our paper.

This folder includes a corrected version of the [HGCN repo](https://github.com/HazyResearch/hgcn) (Chami et al., 2019) together with 9 new synthetic benchmark datasets (the $\text{Tree1111}_\gamma$, for the different values of $\gamma$ mentioned in the paper). Additionally, a python Jupyter notebook is included to show how the features were synthesized for these datasets (giving an example construction for a particular $\gamma$). Exact commands are given below to reproduce the results from the paper. Additionally, an example YAML sweep file for Weights and Biases (wandb) is given as `hgcn/example-sweep-airport-nc.yml` to highlight how we obtained the hyperparameters (primarily for the Euclidean MLP) of our paper.

Obtaining Graph Embeddings for a Graph Dataset           |
:-------------------------:
![Graph Embeddings for a Graph Dataset](https://i.imgur.com/W5aGSeZ.png)|

## HGCN set-up

One need only run `set_env.sh`, as in the original repo.

## Hyperparameter Tuning

All hyperparameters were tuned with Weights and Biases (wandb). An explicit .yaml file with all hyperparameter sweep settings is provided in the hgcn subfolder; the file is "hgcn/example-sweep-airport-nc.yml".

## Main Table 1 results

In this section, we give commands with hyperpameters for direct reproduction of the MLP results in Table 1.

Disease LP:

```
python train.py --act=None --bias=1 --cuda=-1 --dataset=disease_lp --dim=24 --dropout=0.00038556912523335374 --epochs=10000 --gamma=0.1 --grad-clip=None --lr=0.21898633996734412 --lr-reduce-freq=1500 --manifold=Euclidean --model=MLP --momentum=0.8191295002582986 --normalize-feats=0 --num-layers=4 --optimizer=Adam --print-epoch=True --save=0 --save-dir=None --seed=0 --task=lp --weight-decay=0.0010157035973983144 --patience=-1
```

Disease NC:

```
python train.py --act=tanh --bias=1 --cuda=-1 --dataset=disease_nc --dim=32 --dropout=0.1161517854429404 --epochs=1000 --gamma=0.5 --grad-clip=None --lr=0.2336190427982064 --lr-reduce-freq=None --manifold=Euclidean --model=MLP --momentum=0.9507797814298208 --normalize-feats=0 --num-layers=2 --optimizer=Adam --print-epoch=True --save=0 --save-dir=None --seed=0 --task=nc --weight-decay=0.00028123660253562765 --patience=-1
```

Airport LP:

```
python train.py --act=tanh --bias=1 --cuda=-1 --dataset=airport --dim=32 --dropout=0.013250962196331095 --epochs=10000 --gamma=0.5 --grad-clip=None --lr=0.06687543041370096 --lr-reduce-freq=None --manifold=Euclidean --model=MLP --momentum=0.8971286110541203 --normalize-feats=0 --num-layers=3 --optimizer=Adam --print-epoch=True --save=0 --save-dir=None --seed=0 --task=lp --weight-decay=0.00020269210964388252 --patience=-1
```

Airport NC:

```
python train.py --act=tanh --bias=1 --cuda=-1 --dataset=airport --dim=24 --dropout=0.010104339789838268 --epochs=5000 --gamma=0.1 --grad-clip=None --lr=0.05857894169529185 --lr-reduce-freq=750 --manifold=Euclidean --model=MLP --momentum=0.5818657710086917 --normalize-feats=0 --num-layers=4 --optimizer=Adam --print-epoch=True --save=0 --save-dir=None --seed=0 --task=nc --weight-decay=8.075466504904649e-05 --patience=-1
```

Disease-M LP:

```
python train.py --act=tanh --bias=1 --cuda=-1 --dataset=disease_propagation_multi_lp --dim=32 --dropout=0.008858846766604922 --epochs=10000 --gamma=0.1 --grad-clip=None --lr=0.09748485081801048 --lr-reduce-freq=1500 --manifold=Euclidean --model=MLP --momentum=0.6610902352706989 --normalize-feats=0 --num-layers=4 --optimizer=Adam --print-epoch=True --save=0 --save-dir=None --seed=0 --task=lp --weight-decay=0.000929327257066545 --patience=-1
```

## Table 2 and Figure 2 results

In this section we give the commands to reproduce the MLP results on the 9 $\text{Tree1111}_\gamma$ datasets, as well as the commands for the GCN and HyboNet results on Tree1111 ($\gamma=0$).

MLP on $\text{Tree1111}_0$:

```
python train.py --act=relu --bias=1 --cuda=-1 --dataset=tree1111_g00_lp --dim=24 --dropout=0.29512587145564473 --epochs=1000 --gamma=0.9 --grad-clip=None --lr=0.003167159471711093 --lr-reduce-freq=1500 --manifold=Euclidean --model=MLP --momentum=0.9763882226293517 --normalize-feats=0 --num-layers=4 --optimizer=Adam --print-epoch=True --save=0 --save-dir=None --seed=1234 --task=lp --weight-decay=0.0015289383310266966 --patience=-1
```

MLP on $\text{Tree1111}_{0.05}$:

```
python train.py --act=None --bias=0 --cuda=-1 --dataset=tree1111_g005_lp --dim=16 --dropout=0.02406962633183052 --epochs=10000 --gamma=0.1 --grad-clip=None --lr=0.4365052583165098 --lr-reduce-freq=750 --manifold=Euclidean --model=MLP --momentum=0.9950359885284858 --normalize-feats=0 --num-layers=3 --optimizer=Adam --print-epoch=True --save=0 --save-dir=None --seed=0 --task=lp --weight-decay=0.001561244718064682 --patience=-1
```

MLP on $\text{Tree1111}_{0.1}$:

```
python train.py --act=tanh --bias=0 --cuda=-1 --dataset=tree1111_g010_lp --dim=24 --dropout=0.44254170560998174 --epochs=5000 --gamma=0.5 --grad-clip=None --lr=0.007151839739593169 --lr-reduce-freq=750 --manifold=Euclidean --model=MLP --momentum=0.5176262192698449 --normalize-feats=0 --num-layers=2 --optimizer=Adam --print-epoch=True --save=0 --save-dir=None --seed=0 --task=lp --weight-decay=0.0006954873691272352 --patience=-1
```

MLP on $\text{Tree1111}_{0.15}$:

```
python train.py --act=tanh --bias=1 --cuda=-1 --dataset=tree1111_g015_lp --dim=32 --dropout=0.4273602697348596 --epochs=5000 --gamma=0.1 --grad-clip=None --lr=0.019779517176129073 --lr-reduce-freq=750 --manifold=Euclidean --model=MLP --momentum=0.9095524081168604 --normalize-feats=0 --num-layers=2 --optimizer=Adam --print-epoch=True --save=0 --save-dir=None --seed=0 --task=lp --weight-decay=0.00176659214014267 --patience=-1
```

MLP on $\text{Tree1111}_{0.2}$:

```
python train.py --act=tanh --bias=1 --cuda=-1 --dataset=tree1111_g02_lp --dim=32 --dropout=0.3617652291748529 --epochs=10000 --gamma=0.5 --grad-clip=None --lr=0.04684139161181511 --lr-reduce-freq=None --manifold=Euclidean --model=MLP --momentum=0.8946600375985205 --normalize-feats=0 --num-layers=2 --optimizer=Adam --print-epoch=True --save=0 --save-dir=None --seed=0 --task=lp --weight-decay=0.00034137992805340913 --patience=-1
```

MLP on $\text{Tree1111}_{0.4}$:

```
python train.py --act=tanh --bias=1 --cuda=-1 --dataset=tree1111_g04_lp --dim=32 --dropout=0.27895123014999296 --epochs=1000 --gamma=0.5 --grad-clip=None --lr=0.08158050231710905 --lr-reduce-freq=750 --manifold=Euclidean --model=MLP --momentum=0.7446858840368098 --normalize-feats=0 --num-layers=2 --optimizer=Adam --print-epoch=True --save=0 --save-dir=None --seed=0 --task=lp --weight-decay=0.0011687852366054623 --patience=-1
```

MLP on $\text{Tree1111}_{0.6}$:

```
python train.py --act=tanh --bias=1 --cuda=-1 --dataset=tree1111_g06_lp --dim=24 --dropout=0.0025953429837128894 --epochs=5000 --gamma=0.9 --grad-clip=None --lr=0.03674309068617077 --lr-reduce-freq=1500 --manifold=Euclidean --model=MLP --momentum=0.710274508987411 --normalize-feats=0 --num-layers=3 --optimizer=Adam --print-epoch=True --save=0 --save-dir=None --seed=0 --task=lp --weight-decay=0.00032041564549547895 --patience=-1
```

MLP on $\text{Tree1111}_{0.8}$:

```
python train.py --act=tanh --bias=1 --cuda=-1 --dataset=tree1111_g08_lp --dim=32 --dropout=0.04082794728902095 --epochs=5000 --gamma=0.5 --grad-clip=None --lr=0.34590832019942475 --lr-reduce-freq=1500 --manifold=Euclidean --model=MLP --momentum=0.9538015658676154 --normalize-feats=0 --num-layers=2 --optimizer=Adam --print-epoch=True --save=0 --save-dir=None --seed=0 --task=lp --weight-decay=0.0016286404335173002 --patience=-1
```

MLP on $\text{Tree1111}_{1.0}$:

```
python train.py --act=tanh --bias=0 --cuda=-1 --dataset=tree1111_g10_lp --dim=32 --dropout=0.3513462877772794 --epochs=10000 --gamma=0.9 --grad-clip=None --lr=0.0022732908193023316 --lr-reduce-freq=None --manifold=Euclidean --model=MLP --momentum=0.7757167959226898 --normalize-feats=0 --num-layers=3 --optimizer=Adam --print-epoch=True --save=0 --save-dir=None --seed=0 --task=lp --weight-decay=0.001647583275378064 --patience=-1
```

GCN on $\text{Tree1111}_{0}$:

```
python train.py --act=relu --bias=0 --cuda=-1 --dataset=tree1111_g00_lp --dim=32 --dropout=0.4469882260962948 --epochs=1000 --gamma=0.5 --grad-clip=None --lr=0.2382416871345659 --lr-reduce-freq=750 --manifold=Euclidean --model=GCN --momentum=0.6658219653845718 --normalize-feats=0 --num-layers=4 --optimizer=Adam --print-epoch=True --save=0 --save-dir=None --seed=0 --task=lp --weight-decay=0.001222755084943606 --patience=-1
```

HyboNet on $\text{Tree1111}_{0}$:

```
python train.py --act=tanh --bias=0 --cuda=-1 --dataset=tree1111_g00_lp --dim=24 --dropout=0.3654186895693169 --epochs=10000 --gamma=0.9 --grad-clip=None --lr=0.4941337733951095 --lr-reduce-freq=750 --manifold=Lorentz --model=HyboNet --momentum=0.6215759601493149 --normalize-feats=0 --num-layers=4 --optimizer=radam --patience=-1 --print-epoch=True --save=0 --save-dir=None --seed=0 --task=lp --weight-decay=0.0012511827632878463 --patience=-1
```

## Attribution

If you use this code or our results in your research, please cite:

```
@article{Katsman2024SheddingLO,
  title={Shedding Light on Problems with Hyperbolic Graph Learning},
  author={Isay Katsman and Anna Gilbert},
  journal={ArXiv},
  year={2024},
  volume={abs/2411.06688},
  url={https://api.semanticscholar.org/CorpusID:273963557}
}
```
