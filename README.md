# Elastic-Clustering
 Implementation of elastic clustering 

Corresponding paper can be found for free [here](https://arxiv.org/abs/2404.18383), please read for method details.

Several methods exist for teaching robots, with one of the most prominent being Learning from Demonstration (LfD). If a primitive skill is demonstrated, it can be immediately encoded. However, robots must remember multiple skills to be pulled from at will. In order to differentiate demonstrations, we cluster similar primitives together. Similar skills are put in the same cluster, such that if a reproduction of that skill is required, one or more demonstrations can be recalled.

<img src="https://github.com/brenhertel/Elastic-Clustering/blob/main/pictures/RAIL/RAIL_cluster_comparison.png" alt="" width="800"/>

This repository implements the method described in the paper above using Python. All code for clustering can be found in `scripts\elastic_clustering.py` which implements elastic clustering as well as examples. The file `scripts\featurizer.py` implements turning demonstrations into features and clustering, as well as a comparison with agglomerative clustering. If you have any questions, please contact Brendan Hertel (brendan_hertel@student.uml.edu).

If you use the code present in this repository, please cite the following paper:
```
@inproceedings{hertel2024reusable_skills,
  title={A Framework for Learning and Reusing Robotic Skills},
  author={Hertel, Brendan and Tran, Nhu and Elkoudi, Meriem and Azadeh, Reza},
  booktitle={21st International Conference on Ubiquitous Robots (UR)},
  year={2024}
}
```
