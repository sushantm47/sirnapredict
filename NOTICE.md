# Notices and attribution

## Code

Copyright (c) 2026 Sushant M. Released under the MIT License (see `LICENSE`).

## Data

The model in this repository was trained on a published siRNA knockdown dataset:

> Hüsken, D., Lima, W., Guerrier-Takada, C., *et al.* (2005). Design of a genome-wide siRNA
> library using an artificial neural network. *Nature Biotechnology* 23(8), 995–1001.
> https://doi.org/10.1038/nbt1118

**That dataset is not redistributed here.** The repository contains only derived artefacts: the
fitted model coefficients (`model/sirna_model.pkl`) and aggregate evaluation scores
(`model/metrics.json`), neither of which contains or can reconstruct the underlying measurements.
`examples/guides_known.csv` quotes six sequences and their published efficacies for verification.

Users must obtain the dataset themselves from the publication above, or from another distributor
whose licence permits their intended use, and are responsible for complying with those terms.
`scripts/prepare_data.py` converts common published layouts into the format this project expects.

The data concerned is nucleotide sequences and in-vitro assay measurements. It contains no personal
data, no human-subject data, and no clinical data.

## Methods

Design rules and thermodynamic parameters implemented in `src/features.py` come from the published
literature and are cited in the source:

> Reynolds, A., Leake, D., Boese, Q., *et al.* (2004). Rational siRNA design for RNA interference.
> *Nature Biotechnology* 22(3), 326–330.
>
> Khvorova, A., Reynolds, A., Jayasena, S. D. (2003). Functional siRNAs and miRNAs exhibit strand
> bias. *Cell* 115(2), 209–216.
>
> Xia, T., SantaLucia, J., Burkard, M. E., *et al.* (1998). Thermodynamic parameters for an expanded
> nearest-neighbor model. *Biochemistry* 37(42), 14719–14735.

## Independence

This project was built from publicly available sources as coursework for IE-7374 (MLOps),
Northeastern University. It contains no proprietary, confidential or unpublished data or methods
from any employer or third party.

## AI assistance

Portions of this codebase were developed with assistance from an AI coding assistant. All design
decisions, validation choices and reported results were reviewed and verified by the author.
