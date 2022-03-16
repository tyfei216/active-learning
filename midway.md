Batch active learning for drug target gene discovery 

Abstract 

An essential step in early stage drug discovery is target gene validation that serves to assess initial hypotheses about causal associations biological mechanisms and disease pathologies. However, the experimental design space for such genetic experiments is extremely vast and effective methods such as active learning strategies are absolutely necessary. In this project, CRISPR screen datasets are used to evaluate the performance of different algorithms. This task can be viewed as a regression of the effect of a gene being knocked out. By only querying a subset of all genes, we hope we can build a regression model that can accurately predict all possible target genes. Further, we plan to integrate feature selection and multitasking methods into the system to improve performance. 

Introduction:

The discovery and development of new drugs is very challenging with success rate of around 5%. Methods to accelerate this process may potentially save billions of dollars. Among the first steps of drug discovery is target gene discovery which aims to find the related genes for a disease phenotype. However, there are more than 20000 protein-coding genes, thousands of different cell types and numerous environmental conditions. It is simply impossible to do a brute search over such experimental space. Therefore, machine learning methods, such as active learning, could potentially aid in optimally exploring the space of genetic interventions by prioritising experiments that are more likely to yield mechanistic insights of therapeutic relevance. 

The problem can be formalized by the Rubin-Neyman potential outcome framework. A given dataset consists of covariates $X \in \mathbb{R}^p$ with input feature dimensionality $p \in \mathbb{N}$ and treatment descriptors $T \in \mathbb{R}^q$ with treatment descriptor dimensionality $q \in \mathbb{N}^+$ that indicate similarity between interventions. Our aim is to estimate the expectation of the conditional distribution of an unseen outcome $Y_t \in \mathbb{R}$ given observed covariates $X=x$ and intervention $do(T=t)$. $y_t = \mathbb{E}[Y|X=x, do(T=t)]$. In our task, the covariates refers to the cell type and environmental conditions and treatment descriptors refers to the gene we are going to interfere in a experiment. $y$ refers to the effect of the gene being interfered in the cell. 

Data:

In this project, we are going to use the datasets and benchmarks provided by Genedisco. The feature set for genes comes from Achilles and Search Tool for the Retrieval of Interacting Genes/Proteins (STRING) Network Embeddings. 

- The Achilles project generated dependency scores across cancer cell lines by assaying 808 cell lines covering a broad range of tissue types and cancer types (Dempster et al., 2019). The genetic intervention effects are based on interventional CRISPR screens performed across the included cell lines. 
- The STRING (Szklarczyk et al., 2021) database collates known and predicted protein-protein interactions (PPIs) for both physical as well as for functional association. PPI network embeddings could be an informative descriptor of functional gene similarity since proteins that functionally interact with the same network partners may serve similar biological functions. 

The datasets used for evaluation: 

- Interleukin-2 production in primary human T cells. This dataset is based on a genome-wide CRISPR interference (CRISPRi) screen in primary human T cells to uncover the genes regulating the production of Interleukin-2 (IL-2). CRISPRi screens test for loss-of-function genes by reducing their expression levels. The effect is measured by Log fold change in IL-2 normalized read counts. 
- Interferon-$\gamma$ production in primary human T cells. This dataset is based on a genome-wide CRISPR interference (CRISPRi) screen in primary human T cells to understand genes driving production of Interferon-$\gamma$. Interferon-$\gamma$ is a cytokine like interleukin-2. The effected of different genes are also measured by Log fold change in IL-2 normalized read counts. 
- Vulnerability of leukemia cells to NK cells. This genome-wide CRISPR screen was performed in the K562 cell line to identify genes regulating the sensitivity of leukemia cells to cytotoxic activity of primary human NK cells. The effect is measured by Log fold counts of gRNAs in surviving K562 cells (after exposition to NK cells) compared to control (no exposition to NK cells).

Batch Active learning:

​	The experiments are carried out by batch active learning. This means reading out additional values for unexplored interventions $t$ requires a lab experiment and can be costly and time-consuming. Besides, the experiments are carried out in a paralleled manner, which means multiple queries are handled at the same time. Therefore, each time the algorithm will need to provide a list of data points for querying. The performance will be measured by the mean square error between experimentally observed true outcomes $y_t$ and predicted outcomes $\hat{y_t}$. Different batch size and different base regression models will be assessed to give a comprehensive result. 

Multitasking

​	Usually different datasets are evaluated independently. Since different datasets share many common points, it may be better for a single model to handle different datasets at the same time. In this way, informative representations will be shared between different datasets. For example, let a neural network output the regression results on two or more datasets. Another idea is using transfer learning. We may first train the model based on one dataset (all ground truth is known) and then fine tune it on the other dataset with active learning. 

Feature selection 

​	One problem we could have met is that some features might be less related to the outcome. The features given by STRING and Achilles are global features about the genes tested on hundreds of different cell lines. However, the datasets mainly concerns about immune cells and much of the features could be redundant or even causes the model to be overfitted. One idea is that during active learning, we carry out feature selection that turns down the redundant dimensions of the features. In this way, we may be able to pick more informative points for querying and achieve better results. 