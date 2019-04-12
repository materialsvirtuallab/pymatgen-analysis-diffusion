# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.
"""
Created on April 01, 2019
"""

__author__ = "Jimmy Shen"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Jimmy Shen"
__email__ = "jmmshn@lbl.gov"
__date__ = "April 1, 2019"

from pprint import pprint
from collections import defaultdict
import math
from copy import deepcopy
import os
import logging
import sys
from monty.serialization import loadfn
from pymatgen.core.periodic_table import Element, Specie
from pymatgen.core.structure import Composition, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import Structure, PeriodicSite
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import MinimumDistanceNN
import operator
import numpy as np
import networkx as nx
from itertools import starmap
from pymatgen_diffusion.neb.pathfinder import MigrationPath


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


class FullPathMapper:
    """
    Find all hops in a given crystal structure using the StructureGraph.
    Each hop is an edge in the StructureGraph object and each node is a position of the migrating species in the structure
    The equivalence of the hops is checked using the MigrationPath.__eq__ funciton.
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

        # Generate the graph edges between these all the sites
        self.s_graph = StructureGraph.with_local_env_strategy(
            self.only_sites,
            MinimumDistanceNN(
                cutoff=max_path_length,
                get_all_sites=True))  # weights in this graph are the distances
        self.s_graph.set_node_attributes()

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
        edge['fpos'] = f_site.frac_coords
        edge['ipos_cart'] = np.dot(i_site.frac_coords,
                                   self.only_sites.lattice.matrix)
        edge['fpos_cart'] = np.dot(f_site.frac_coords,
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
        for u, v, d in enumerate(fpm.s_graph.graph.edges(data=True)):
            if d['hop_label'] == taget_label:
                d.update(data)
