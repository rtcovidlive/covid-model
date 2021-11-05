# Model powering rt.live
This repository contains the code of the data processing and modeling behind https://rt.live.

__Because this code is running *in production*, the maintainers of this repository are *very* conservative about merging any PRs.__

## Application to Other Countries
We have learned that it takes continuous attention to keep running the model. This is mostly due to data quality issues that are __best solved with local domain knowledge__.

In other words, the maintainers behind this repo and http://rt.live don't currently have the resources to ensure high-quality analyses for other countries.

However, we encourage you to apply and improve the model for your country!

## Contributing
We are open to PRs that address aspects of the code or model that generalize across borders.
For example on the topics of:
+ docstrings (NumPy-style), 
+ testing
+ robustness against data outliers
+ computational performance
+ model insight

## Citing
To reference this project in a scientific article:
```
Kevin Systrom, Thomas Vladek and Mike Krieger. Rt.live (2020). GitHub repository, https://github.com/rtcovidlive/covid-model
```
or with the respective BibTeX entry:
```
@misc{rtlive2020,
  author = {Systrom, Kevin and Vladek, Thomas and Krieger, Mike},
  title = {Rt.live},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rtcovidlive/covid-model}},
  commit = {...}
}
```
