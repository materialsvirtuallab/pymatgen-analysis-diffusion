# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.
"""
Created on April 01, 2019
"""

__author__ = "Jimmy Shen"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Jimmy Shen"
__email__ = "jmmshn@lbl.gov"
__date__ = "April 11, 2019"

from copy import deepcopy
import logging
from pymatgen.io.vasp import VolumetricData
from pymatgen.core.structure import Composition
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import Structure, PeriodicSite
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, generate_full_symmops
from pymatgen.analysis.local_env import MinimumDistanceNN
import operator
import numpy as np
import networkx as nx
from itertools import starmap
from pymatgen_diffusion.neb.pathfinder import MigrationPath
from monty.json import MSONable

logger = logging.getLogger(__name__)


def generic_groupby(list_in, comp=operator.eq):
    """
    Group a list of unsortable objects

    Args:
        list_in: A list of generic objects
        comp: (Default value = operator.eq) The comparator

    Returns:
        [int] list of labels for the input list

    """
    list_out = [None] * len(list_in)
    label_num = 0
    for i1, ls1 in enumerate(list_out):
        if ls1 is not None:
            continue
        list_out[i1] = label_num
        for i2, ls2 in list(enumerate(list_out))[i1 + 1:]:
            if comp(list_in[i1], list_in[i2]):
                list_out[i2] = list_out[i1]
        label_num += 1
    return list_out


class FullPathMapper(MSONable):
    """
    Find all hops in a given crystal structure using the StructureGraph.
    Each hop is an edge in the StructureGraph object and each node is a position of the migrating species in the
    structure
    The equivalence of the hops is checked using the MigrationPath.__eq__ funciton.
    The funtions here are reponsible for distinguishing the individual hops and analysis
    """

    def __init__(self,
                 structure,
                 migrating_specie,
                 max_path_length=10,
                 symprec=0.1,
                 vac_mode=False):
        """
        Args:
            structure: Input structure that contains all sites.
            migrating_specie (Specie-like): The specie that migrates. E.g.,
                "Li".
            max_path_length (float): Maximum length of NEB path in the unit
                of Angstrom. Defaults to None, which means you are setting the
                value to the min cutoff until finding 1D or >1D percolating paths.
            symprec (float): Symmetry precision to determine equivalence.
        """
        self.structure = structure
        self.migrating_specie = get_el_sp(migrating_specie)
        self.symprec = symprec
        self.a = SpacegroupAnalyzer(self.structure, symprec=self.symprec)
        self.symm_structure = self.a.get_symmetrized_structure()
        self.only_sites = self.get_only_sites()
        self.unique_hops = None

        # Generate the graph edges between these all the sites
        self.s_graph = StructureGraph.with_local_env_strategy(
            self.only_sites,
            MinimumDistanceNN(
                cutoff=max_path_length,
                get_all_sites=True))  # weights in this graph are the distances
        self.s_graph.set_node_attributes()

    # TODO add classmethod for creating the FullPathMapper from the charge density

    def get_only_sites(self):
        """
        Get a copy of the structure with only the sites

        Args:

        Returns:
          Structure: Structure with all possible migrating ion sites

        """
        migrating_ion_sites = list(
            filter(
                lambda site: site.species == Composition(
                    {self.migrating_specie: 1}), self.structure.sites))
        return Structure.from_sites(migrating_ion_sites)

    def _get_pos_and_migration_path(self, u, v, w):
        """
        insert a single MigrationPath object on a graph edge
        Args:
          u (int): index of initial node
          v (int): index of final node
          w (int): index for multiple edges that share the same two nodes

        """
        edge = self.s_graph.graph[u][v][w]
        i_site = self.only_sites.sites[u]
        f_site = PeriodicSite(
            self.only_sites.sites[v].species,
            self.only_sites.sites[v].frac_coords + np.array(edge['to_jimage']),
            lattice=self.only_sites.lattice)
        # Positions might be useful for plotting
        edge['ipos'] = i_site.frac_coords
        edge['epos'] = f_site.frac_coords
        edge['ipos_cart'] = np.dot(i_site.frac_coords,
                                   self.only_sites.lattice.matrix)
        edge['epos_cart'] = np.dot(f_site.frac_coords,
                                   self.only_sites.lattice.matrix)

        edge['hop'] = MigrationPath(i_site, f_site, self.symm_structure)

    def populate_edges_with_migration_paths(self):
        """
        Populate the edges with the data for the Migration Paths
        """
        list(
            starmap(self._get_pos_and_migration_path,
                    self.s_graph.graph.edges))

    def group_and_label_hops(self):
        """
        Group the MigrationPath objects together and label all the symmetrically equlivaelnt hops with the same label
        """
        hops = [(g_index, val) for g_index, val in nx.get_edge_attributes(
            self.s_graph.graph, "hop").items()]
        labs = generic_groupby(hops, comp=lambda x, y: x[1] == y[1])
        new_attr = {
            g_index: {
                'hop_label': labs[edge_index]
            }
            for edge_index, (g_index, _) in enumerate(hops)
        }
        nx.set_edge_attributes(self.s_graph.graph, new_attr)
        return new_attr

    def get_unique_hops_dict(self):
        """
        Get the list of the unique objects
        Returns:
            dictionary {label : MigrationPath}

        """
        self.unique_hops = {
            d['hop_label']: d['hop']
            for u, v, d in self.s_graph.graph.edges(data=True)
        }

    def add_data_to_similar_edges(self, taget_label, data=dict()):
        """
        Insert data to all edges with the same label
        """
        for u, v, d in self.s_graph.graph.edges(data=True):
            if d['hop_label'] == taget_label:
                d.update(data)


class ComputedEntryPath(FullPathMapper):
    """
    Generate the full migration network using computed entires for intercollation andvacancy limits
    - Map the relaxed sites of a material back to the empty host lattice
    - Apply symmetry operations of the empty lattice to obtain the other positions of the intercollated atom
    - Get the symmetry inequivalent hops
    - Get the migration barriers for each inequivalent hop
    """

    def __init__(self,
                 base_struct_entry,
                 single_cat_entries,
                 migrating_specie,
                 base_aeccar=None,
                 max_path_length=4,
                 ltol=0.2,
                 stol=0.3,
                 full_sites_struct=None,
                 angle_tol=5):
        """
        Pass in a entries for analysis

        Args:
          base_struct_entry: the structure without a working ion for us to analyze the migration
          single_cat_entries: list of structures containing a single cation at different positions
          base_aeccar: Chgcar object that contains the AECCAR0 + AECCAR2 (Default value = None)
          migration_specie: a String symbol or Element for the cation. (Default value = 'Li')
          ltol: parameter for StructureMatcher (Default value = 0.2)
          stol: parameter for StructureMatcher (Default value = 0.3)
          angle_tol: parameter for StructureMatcher (Default value = 5)
        """

        self.single_cat_entries = single_cat_entries
        self.base_struct_entry = base_struct_entry
        self.base_aeccar = base_aeccar
        self.migrating_specie = migrating_specie
        self._tube_radius = 0
        self.sm = StructureMatcher(
            comparator=ElementComparator(),
            primitive_cell=False,
            ignored_species=[migrating_specie],
            ltol=ltol,
            stol=stol,
            angle_tol=angle_tol)

        logger.debug('See if the structures all match')
        fit_ents = []
        if full_sites_struct:
            self.full_sites = full_sites_struct
            self.base_structure_full_sites = self.full_sites.copy()
            self.base_structure_full_sites.sites.extend(
                self.base_struct_entry.structure.sites)
        else:
            for ent in self.single_cat_entries:
                if self.sm.fit(self.base_struct_entry.structure, ent.structure):
                    fit_ents.append(ent)
            self.single_cat_entries = fit_ents

            self.translated_single_cat_entries = list(
                map(self.match_ent_to_base, self.single_cat_entries))
            self.full_sites = self.get_full_sites()
            self.base_structure_full_sites = self.full_sites.copy()
            self.base_structure_full_sites.sites.extend(
                self.base_struct_entry.structure.sites)

        # Initialize
        super(ComputedEntryPath, self).__init__(
            structure=self.base_structure_full_sites,
            migrating_specie=migrating_specie,
            max_path_length=max_path_length,
            symprec=0.1,
            vac_mode=False)

        self.populate_edges_with_migration_paths()
        self.group_and_label_hops()
        self.get_unique_hops_dict()
        if base_aeccar:
            self._setup_grids()

    def match_ent_to_base(self, ent):
        """
        Transform the structure of one entry to match the base structure

        Args:
          ent:

        Returns:
          ComputedStructureEntry: entry with modified structure

        """
        new_ent = deepcopy(ent)
        new_struct = self.sm.get_s2_like_s1(self.base_struct_entry.structure,
                                            ent.structure)
        new_ent.structure = new_struct
        return new_ent

    def get_full_sites(self):
        """
        Get each group of symmetry inequivalent sites and combine them

        Args:

        Returns: a Structure with all possible Li sites, the enregy of the structure is stored as a site property

        """
        res = []
        for itr in self.translated_single_cat_entries:
            sub_site_list = get_all_sym_sites(itr, self.base_struct_entry,
                                              self.migrating_specie)
            # ic(sub_site_list._sites)
            res.extend(sub_site_list._sites)
        res = Structure.from_sites(res)
        # ic(res)
        if len(res) > 1:
            res.merge_sites(tol=1.0, mode='average')
        # ic(res)
        return res

    def _setup_grids(self):
        """Populate the internal varialbes used for defining the grid points in the charge density analysis"""

        def _shift_grid(vv):
            """
            Move the grid points by half a step so that they sit in the center

            Args:
              vv: equally space grid points in 1-D

            """
            step = vv[1] - vv[0]
            vv += step / 2.

        # set up the grid
        aa = np.linspace(
            0, 1, len(self.base_aeccar.get_axis_grid(0)), endpoint=False)
        bb = np.linspace(
            0, 1, len(self.base_aeccar.get_axis_grid(1)), endpoint=False)
        cc = np.linspace(
            0, 1, len(self.base_aeccar.get_axis_grid(2)), endpoint=False)
        # move the grid points to the center
        _shift_grid(aa)
        _shift_grid(bb)
        _shift_grid(cc)

        # mesh grid for each unit cell
        AA, BB, CC = np.meshgrid(aa, bb, cc, indexing='ij')

        # should be using a mesh grid of 5x5x5 (using 3x3x3 misses some fringe cases)
        # but using 3x3x3 is much faster and only crops the cyliners in some rare case
        # if you keep the tube_radius small then this is not a big deal
        IMA, IMB, IMC = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1],
                                    indexing='ij')

        # store these
        self._uc_grid_shape = AA.shape
        self._fcoords = np.vstack([AA.flatten(), BB.flatten(), CC.flatten()]).T
        self._images = np.vstack([IMA.flatten(),
                                  IMB.flatten(),
                                  IMC.flatten()]).T

    def _get_chg_between_sites_tube(self,
                                    migration_path,
                                    mask_file_seedname=None):
        """
        Calculate the amount of charge that a migrating ion has to move through in order to complete a hop

        Args:
            migration_path: MigrationPath object that represents a given hop
            mask_file_seedname(string): seedname for output of the migration path masks (for debugging and
                visualization) (Default value = None)

        Returns:
            float: The total charge density in a tube that connects two sites of a given edges of the graph

        """
        try:
            self._tube_radius
        except NameError:
            logger.error(
                "The radius of the tubes for charge analysis need to be defined first."
            )
        ipos = migration_path.isite.frac_coords
        epos = migration_path.esite.frac_coords

        cart_ipos = np.dot(ipos, self.base_aeccar.structure.lattice.matrix)
        cart_epos = np.dot(epos, self.base_aeccar.structure.lattice.matrix)
        pbc_mask = np.zeros(self._uc_grid_shape, dtype=bool).flatten()
        for img in self._images:
            grid_pos = np.dot(self._fcoords + img,
                              self.base_aeccar.structure.lattice.matrix)
            proj_on_line = np.dot(grid_pos - cart_ipos,
                                  cart_epos - cart_ipos) / (
                               np.linalg.norm(cart_epos - cart_ipos))
            dist_to_line = np.linalg.norm(
                np.cross(grid_pos - cart_ipos, cart_epos - cart_ipos) /
                (np.linalg.norm(cart_epos - cart_ipos)),
                axis=-1)

            mask = (proj_on_line >= 0) * (proj_on_line < np.linalg.norm(
                cart_epos - cart_ipos)) * (dist_to_line < self._tube_radius)
            pbc_mask = pbc_mask + mask
        pbc_mask = pbc_mask.reshape(self._uc_grid_shape)

        if mask_file_seedname:
            mask_out = VolumetricData(
                structure=self.base_aeccar.structure.copy(),
                data={'total': self.base_aeccar.data['total']})
            mask_out.structure.insert(0, "X", ipos)
            mask_out.structure.insert(0, "X", epos)
            mask_out.data['total'] = pbc_mask
            isym = self.symm_structure.wyckoff_symbols[migration_path.iindex]
            esym = self.symm_structure.wyckoff_symbols[migration_path.eindex]
            mask_out.write_file('{}_{}_{}_tot({:0.2f}).vasp'.format(
                mask_file_seedname, isym, esym, mask_out.data['total'].sum()))

        return self.base_aeccar.data['total'][pbc_mask].sum(
        ) / self.base_aeccar.ngridpts / self.base_aeccar.structure.volume

    def populate_edges_with_chg_density_info(self, tube_radius=1):
        self._tube_radius = tube_radius
        for k, v in self.unique_hops.items():
            chg_tot = self._get_chg_between_sites_tube(v)
            self.add_data_to_similar_edges(k, {'chg_total': chg_tot})


def get_all_sym_sites(ent, base_struct_entry, migrating_specie, stol=1.0, atol=10):
    """
    Return all of the symmetry equivalent sites by applying the symmetry operation of the empty structure

    Args:
        ent(ComputedStructureEntry): that contains cation
        migrating_species(string or Elment):

    Returns:
        Structure: containing all of the symmetry equivalent sites

    """
    migrating_specie_el = get_el_sp(migrating_specie)
    sa = SpacegroupAnalyzer(
        base_struct_entry.structure,
        symprec=stol,
        angle_tolerance=atol)
    # start with the base structure but empty
    host_allsites = base_struct_entry.structure.copy()
    host_allsites.remove_species(host_allsites.species)
    pos_Li = list(
        filter(lambda isite: isite.species_string == migrating_specie_el.name,
               ent.structure.sites))
    for isite in pos_Li:
        host_allsites.insert(
            0,
            migrating_specie_el.name,
            np.mod(isite.frac_coords, 1),
            properties={'inserted_energy': ent.energy})
    # base_ops = sa.get_space_group_operations()
    # all_ops = generate_full_symmops(base_ops, tol=1.0)
    for op in sa.get_space_group_operations():
        logger.debug(f'{op}')
        struct_tmp = host_allsites.copy()
        struct_tmp.apply_operation(symmop=op, fractional=True)
        for isite in struct_tmp.sites:
            if isite.species_string == migrating_specie_el.name:
                logger.debug(f'{op}')
                host_allsites.insert(
                    0,
                    migrating_specie_el.name,
                    np.mod(isite.frac_coords, 1),
                    properties={'inserted_energy': ent.energy})
                host_allsites.merge_sites(
                    mode='average'
                )  # keeps only one position but average the properties

    return host_allsites
