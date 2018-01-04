from pymatgen.analysis.transition_state import NEBAnalysis

neb = NEBAnalysis.from_dir("/Users/hanmeiTang/research/"
                           "lina_nasicon/data_col/10_ng/4_neb")
p = neb.get_plot(normalize_rxn_coordinate=True, label_barrier=True)
p.show()

from pymacy.neb.io import plot_barrier

# Modify the mobile_specie_index
p = plot_barrier(neb_directory="/Users/hanmeiTang/research/"
                           "lina_nasicon/data_col/10_ng/4_neb",
             mobile_specie_index=4, label_barrier=True);

p.show()
