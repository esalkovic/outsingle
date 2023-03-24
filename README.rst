OutSingle - A Python tool for finding outliers in RNA-Seq gene expression count data using SVD/OHT
==================================================================================================

OutSingle has been tested on Windows (11).
Note that OutSingle is still in
alpha stage,
so encountering bugs while
running it is expected.
If you use OutSingle in your research
you can cite our paper::

Edin Salkovic, Mohammad Amin Sadeghi, Abdelkader Baggag, Ahmed Gamal Rashed Salem, Halima Bensmail, OutSingle: A Novel Method of Detecting and Injecting Outliers in RNA-seq Count Data Using the Optimal Hard Threshold for Singular Values, Bioinformatics, 2023;, btad142, https://doi.org/10.1093/bioinformatics/btad142

And please also check our other older models (both have related published papers)::

https://github.com/esalkovic/outpyr

https://github.com/esalkovic/outpyrx

Installation
------------
The recommended way is to just clone/download
the repository and open a terminal/CMD.exe,
cd/chdir to the downloaded directory and
run the following command
(preferably inside of a
virtualenv virtual environment)::

  pip install -r requirements.txt

It might take some time to install all dependencies.

Note that OutSingle comes with an optht.py file copied
from https://github.com/erichson/optht and its licence
is included in this repository inside the file optht_LICENSE.

Usage
-----
We recommend that you create a
separate workspace
directory for every data file that you
process as OutSingle's sub-tools will create
some files and folders inside the
directory where the file you run it on
is located.

You can run the various OutSingle sub-tools
on your data as follows::

 python fast_zscore_estimation.py some-dataset/some-dataset.csv

That will estimate a simple z-score for the dataset.
The only parameter that you have to supply
to ``fast_zscore_estimation.py`` is the path to the dataset file
(here it's ``some-dataset/some-dataset.csv``).
The dataset file should be a tab-separated
pandas-compatible CSV file containing
gene expression counts.
Its index (first column) should
contain names of genes,
while its columns header (first row)
should contain the names of samples
that the counts were sequenced from.
Other cells should contain
integer count data.

After z-score estimation, you can calculate the final
OutSingle score with::

 python optht_svd_zs.py some-dataset/some-dataset-fzse-zs.csv
 
That will produce a file ``some-dataset/some-dataset-fzse-zs-svd-optht-zs.csv``
which can be considered as the final OutSingle "outlierness" score
for the dataset.

You can also inject artificial outliers that are masked by
confounding effects (present in the data) with::

 python inject_outliers_fzse_pysvdcc.py some-dataset/some-dataset.csv

That command will produce several files with artificial outliers
(i.e., original data + outliers). These are CSV files but with a ".txt"
extension.
You can edit that script to change some parameters of the injection
process: frequency, magnitudes of outliers, whether the outliers
have only large positive z-scores, only negative, or both (50%/50%).

As an example the tool might produce the following file::

 some-dataset/some-dataset-wo-f1-b-z6.00.txt

``-f1-`` indicates that the frequency of outliers is 1 per sample,
``-b-`` indicates that both outliers with positive and negative z-scores
are present in the data, and ``-z6.00`` indicates that the z-score magnitude
of the outliers is 6.

The tool will also produce corresponding outlier mask (omask) files which
are CSV files (again ending with ".txt")
containing information about the exact "coordinates"
of injected outliers' positions.
Both types of CSV files contain data matrices that are of the same "dimension"
as the data matrix of some-dataset.csv, and the outlier mask files
have all zeros except the "coordinates" of outilers which are 1 or -1
(depending on the z-score of the outlier).

After injecting outliers, you can run the above two tools on files
containing artificial outliers to produce
an OutSingle score for them. E.g.::

 python fast_zscore_estimation.py some-dataset/some-dataset-wo-f1-b-z6.00.txt
 python optht_svd_zs.py some-dataset/some-dataset-wo-f1-b-z6.00-fzse-zs.txt

That will produce a ``some-dataset/some-dataset-wo-f1-b-z6.00-fzse-zs-svd-optht-zs.csv``
OutSingle score file.

Further info
------------
You can check OutPyR's documentation regarding how you can obtain a publicly available
dataset file on which you can apply OutSingle:
https://github.com/esalkovic/outpyr
