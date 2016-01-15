# Pymatgen-diffusion

This is an add-on to pymatgen for diffusion analysis that is developed
by the Materials Virtual Lab. Note that it relies on pymatgen for structural 
manipulations, file io, and preliminary analyses. In particular, the 
pymatgen.analysis.diffusion_analyzer is used heavily. The purpose of this add-on
is to provide other diffusion analyses, using trajectories extracted using the
DiffusionAnalyzer class.

# Features (non-exhaustive!)

1. Van-Hove analysis
2. Probability density
3. Clustering (e.g., k-means with periodic boundary conditions).

# Acknowledgements

This code is funded by the National Science Foundation’s Designing Materials
to Revolutionize and Engineer our Future (DMREF) program under Grant No. 
1436976.
