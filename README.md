## Overview
This repository contains the code used for the paper [Why Do Equally Weighted Portfolios Beat Value-Weighted Ones?](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4280394) by Swade et al. (2023). Please cite this paper if you are using the code:
```
@article{swade2023,
  title={Why Do Equally Weighted Portfolios Beat Value-Weighted Ones?},
  author={Swade, Alexander and Nolte, Sandra and Shackleton, Mark B and Lohre, Harald},
  journal={Journal of Portfolio Management},
  year={2023},
  volume={49},
  number={5},
  pages={167--187},
}
```
The repository is structured as follows:
- [configs](https://github.com/smalswad/equalweighting/tree/main/configs) contains a yaml file with configurations for the base setup of the paper. Store config files in a separate folder following your current working directory '.../cwd/configs/'. 
- [equalweighting](https://github.com/smalswad/equalweighting/tree/main/equalweighting) contains the three separate files for the data loading, the calculation process and the data visualization.
- [data]() contains the used data by WRDS, Bloomberg, K. French and other sources. Please note, I do not have the rights to share any data, but a detailed overview of the used time series is given in the paper.
- [main.py](https://github.com/smalswad/equalweighting/blob/main/main.py) calls the different steps of the calculations and saves results afterwards.

See furhter research papers to related topics at [SSRN](https://papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=3837762).
