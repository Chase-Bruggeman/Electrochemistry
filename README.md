For the electrochemist interested in alternating current voltammetry (ACV)

Two text files of data accompany two python files for interpreting the data.
The file `fraccalc` is a small python package for computing fractional integrals/derivatives of any order, and for baselining functions.
The files `sample_ACV_data` and `sample_CV_data` are text files (CV = cyclic voltammetry).
The file `sample_python_script_for_ACV_fitting` is exactly what it says it is. It uses both sample data files.
The code and data accompany Chase Bruggeman's PhD dissertation (Chapter 2), completed at Michigan State University in Spring 2024.

The file `to_precision.py` allows numbers to be printed with an arbitrary number of sig figs in different notations (standard, scientific, etc.) The package was written by:
William Rusnack github.com/BebeSparkelSparkel linkedin.com/in/williamrusnack williamrusnack@gmail.com
Eric Moyer github.com/epmoyer eric@lemoncrab.com
Thomas Hladish https://github.com/tjhladish tjhladish@utexas.edu

The file `voltammetry_paper_code_APYA.py` can compute an arbitrary number of harmonics for an ACV experiment. The current for several first order mechanisms with one or two step charge transfer may be computed, and cell resistance and double layer charging current may be accounted for. The fundamental harmonic for second order mechanisms may also be computed. The code generates figures too. The `_v2` file includes minor improvements and generates a figure to view several harmonics at once.
