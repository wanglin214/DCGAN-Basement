# DCGAN-Basement

## What is this repository for?
Generation Maps of Basement by DCGAN

## Requirements:
Hardware requirements: NVIDIA GPU ;Program language: Python 3.11 (Pytorch 2.1);Software required: PyCharm, Anaconda

Data requirements: the only need data are basement maps from real-world basins, here we put some samples in the data file just fot test

## how to get start?

1 the codes are used to generate basement map according to the real-world basins.

2 the real maps are included in the data file; the model file DCGAN.py is included in the model file.

3 use trainMod.py to train the DCGAN to reach a set of parameters of Generator and Discriminator which are included in the models file.

4 use testMod.py to generate new maps of basement depth, the output of generated maps are also included in the data file.

5 the related functions are included in the utils file.

## the diagram of DCGAN for basement depth generating and DL architecture are presented as follow:

![DCGAN示意图](https://github.com/user-attachments/assets/d4b5272a-b93f-4ec9-8b70-10b6cb5d7210)

![DCGAN示意图GD](https://github.com/user-attachments/assets/657c2bf6-da99-4a9b-bbb7-e77227f6eadb)
