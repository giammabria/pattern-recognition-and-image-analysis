# Machine Learning for Pattern-recognition-and-image-analysis

## Introduction

This reporsitory summarizes the methodologies, preprocessing techniques, validations, and comparative evaluations conducted during different laboratory sessions for the course "Pattern Recognition and Image Analysis." The analysis was carried out using a subset of the CIFAR-10 dataset, consisting of three classes: 'airplane', 'bird', and 'horse'. Each image was represented by a 256-dimensional Histogram of Gradient (HoG) feature vector. The aim was to evaluate and compare various supervised learning algorithms.

## Project Architecture

The structure is crafted to guide every data science professional and enthusiast through a streamlined workflow, right from raw data ingestion to deployment of the final model API. The project's folder structure would look like this:

```bash
.
├── data
│   ├── CIFAR10
│   └── video_data
│
├── images
│
├── models
│
├── notebooks
│   ├── ML0_introduction.ipynb
│   ├── ML1_Nearest_Neighbours_and_Decision_Trees.ipynb
│   ├── ML2_Neural_Networks.ipynb
│   ├── ML3_SVM.ipynb
│   ├── PR1_Expert_System.ipynb
│   ├── PR2_Feature_Space.ipynb
│   ├── PR3_Points_Neighbourhoods_and_Clustering.ipynb
│   └── PR4_Video_and_Matching.ipynb
│
├── utils
│   └── tools.py
│
├── Report.pdf
│
└── README.md
