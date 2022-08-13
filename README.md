# SimEnhancedGCL
This is the code base for the paper "Enhancing Graph Contrastive Learning with Node Similarity".

## DataSet Statistics
| Dataset  | #Nodes | #Edges | #Features | #Classes |		
| :---: | :---: | :---: | :---: | :---: |		
| Cora  | 2,708 | 5,429 | 1,433 | 7 |		
| Citeseer  | 3,327 | 4,732 | 3,703 | 6 |		
| Pubmed  | 19,717 | 44,338 | 500 | 3 |		
| DBLP  | 17,716 | 105,734 | 1,639 | 4 |		
| Amazon-Computers  | 13,752 | 245,861 | 767| 10 | 
| Amazon-Photo | 7,650 | 119,081 | 745 | 8 | 
| Wiki-CS  | 11,701 | 216,123 | 300 | 10 |
| Coauthor-CS  | 18,333 | 81,894 | 6,805 | 15 |

## Reference
The Simlarity Enhanced Models are developed based on:\
GRACE： https://github.com/CRIPAC-DIG/GRACE \
GCA: https://github.com/CRIPAC-DIG/GCA \
Graph-MLP: https://github.com/yanghu819/Graph-MLP 

## Running
Please use the commands in the data_name_run.sh  to reproduce the reported results for the corresponding datasets;\
e.g., To obtain Graph-MLP+ performance for the Cora Dataset, please run the commands in GraphMLPPlus/cora_run.sh
