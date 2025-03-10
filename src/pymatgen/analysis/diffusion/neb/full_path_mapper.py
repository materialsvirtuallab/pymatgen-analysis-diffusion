"""Migration Graph Analysis."""

from __future__ import annotations

__author__ = "Jimmy Shen"
__copyright__ = "Copyright 2019, The Materials Project"
__maintainer__ = "Jimmy Shen"
__email__ = "jmmshn@lbl.gov"
__date__ = "April 11, 2019"

import itertools
import logging
import operator
from copy import deepcopy
from itertools import starmap
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from monty.json import MSONable

from pymatgen.analysis.diffusion.neb.pathfinder import ChgcarPotential, MigrationHop, NEBPathfinder
from pymatgen.analysis.diffusion.neb.periodic_dijkstra import get_optimal_pathway_rev, periodic_dijkstra
from pymatgen.analysis.diffusion.utils.parse_entries import process_entries
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import MinimumDistanceNN, NearNeighbors
from pymatgen.core import Composition, PeriodicSite, Structure
from pymatgen.io.vasp import VolumetricData
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.structure import SymmetrizedStructure

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
    from pymatgen.util.typing import SpeciesLike

logger = logging.getLogger(__name__)


def generic_groupby(list_in: list, comp: Callable = operator.eq) -> list:
    """
    Group a list of unsortable objects.

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
        list_out[i1] = label_num  # type: ignore
        for i2, _ls2 in list(enumerate(list_out))[i1 + 1 :]:
            if comp(list_in[i1], list_in[i2]):
                if list_out[i2] is None:
                    list_out[i2] = list_out[i1]
                else:
                    list_out[i1] = list_out[i2]
                    label_num -= 1
        label_num += 1
    return list_out


class MigrationGraph(MSONable):
    """
    A python object for handling the migratrion graph of a given base
    structure and mobile specie sites within that base structure.

    Each mobile site is a node in the migration graph and hops
    (between sites) are edges. The migration graph object uses
    the Structure Graph from pymatgen.
    """

    def __init__(
        self,
        structure: Structure,
        m_graph: StructureGraph,
        symprec: float = 0.1,
        vac_mode: bool = False,
    ) -> None:
        """
        Construct the MigrationGraph object using a potential_field will
        all mobile sites occupied. A potential_field graph is generated by
        connecting all sites as specified by the migration graph.
        The sites are decorated with Migration graph objects and then grouped
        together based on their equivalence.

        Args:
            structure: Structure with base framework and mobile sites.
             When used with structure_is_base = True, only the base framework
             structure, does not contain any migrating sites.
            m_graph: The StructureGraph object that defines the
             migration network
            symprec (float): Symmetry precision to determine equivalence
             of migration events
            vac_mode (Bool): indicates whether vacancy mode should be used
        """
        self.structure = structure
        self.m_graph = m_graph
        self.symprec = symprec
        self.vac_mode = vac_mode
        if self.vac_mode:
            raise NotImplementedError("Vacancy mode is not yet implemented")
        # Generate the graph edges between these all the sites
        self.m_graph.set_node_attributes()  # popagate the sites properties to the graph nodes
        # For poperies like unique_hops we might be interested in modifying them after creation
        # So let's not convert them into properties for now.  (Awaiting rewrite once the usage becomes more clear.)
        self._populate_edges_with_migration_hops()
        self._group_and_label_hops()

    @property
    def only_sites(self) -> Structure:
        """A structure that only contains the migrating species."""
        return self.m_graph.structure

    @property
    def host_structure(self) -> Structure:
        """A structure that only contains the non-migrating species."""
        host_struct = self.structure.copy()
        rm_sites = set()
        for isite in self.only_sites:
            neighbors_ = host_struct.get_neighbors_in_shell(isite.coords, r=0.0, dr=0.05, include_index=True)
            if len(neighbors_) == 0:
                continue
            for n_ in neighbors_:
                rm_sites.add(n_.index)
        host_struct.remove_sites(list(rm_sites))
        return host_struct

    @property
    def symm_structure(self) -> SymmetrizedStructure:
        """The symmetrized structure with the present item's symprec value."""
        a = SpacegroupAnalyzer(self.structure, symprec=self.symprec)
        sym_struct = a.get_symmetrized_structure()
        if not isinstance(sym_struct, SymmetrizedStructure):
            raise RuntimeError("Symmetrized structure could not be generated.")
        return sym_struct

    @property
    def unique_hops(self) -> dict:
        """The unique hops dictionary keyed by the hop label."""
        # reversed so that the first instance represents the group of distinct hops
        ihop_data = list(reversed(list(self.m_graph.graph.edges(data=True))))
        for u, v, d in ihop_data:
            d["iindex"] = u
            d["eindex"] = v
            d["hop_distance"] = d["hop"].length
        return {d["hop_label"]: d for u, v, d in ihop_data}

    @classmethod
    def with_base_structure(cls, base_structure: Structure, m_graph: StructureGraph, **kwargs) -> MigrationGraph:
        """
        Args:
            base_structure: base framework structure that does not contain any migrating sites.
            m_graph: The StructureGraph object that defines the migration network.
            **kwargs: Passthrough for kwargs.

        Returns:
            A constructed MigrationGraph object
        """
        sites = m_graph.structure.sites + base_structure.sites
        structure = Structure.from_sites(sites)
        return cls(structure=structure, m_graph=m_graph, **kwargs)

    @classmethod
    def with_local_env_strategy(
        cls, structure: Structure, migrating_specie: str, nn: NearNeighbors, **kwargs
    ) -> MigrationGraph:
        """
        Using a specific nn strategy to get the connectivity graph between all the migrating ion sites.

        Args:
            structure: Input structure that contains all sites.
            migrating_specie: The specie that migrates. E.g. "Li".
            nn: The specific local environment object used to connect the migrating ion sites.
            **kwargs: Passthrough for kwargs.

        Returns:
            A constructed MigrationGraph object
        """
        only_sites = get_only_sites_from_structure(structure, migrating_specie)
        migration_graph = StructureGraph.with_local_env_strategy(only_sites, nn)
        return cls(structure=structure, m_graph=migration_graph, **kwargs)

    @classmethod
    def with_distance(
        cls,
        structure: Structure,
        migrating_specie: SpeciesLike,
        max_distance: float,
        **kwargs,
    ) -> MigrationGraph:
        """
        Using a specific nn strategy to get the connectivity graph between all the migrating ion sites.

        Args:
            structure: Input structure that contains all sites.
            migrating_specie: The specie that migrates. E.g. "Li".
            max_distance: Maximum length of NEB path in the unit
                of Angstrom. Defaults to None, which means you are setting the
                value to the min cutoff until finding 1D or >1D percolating paths.
            **kwargs: Passthrough for kwargs.

        Returns:
            A constructed MigrationGraph object
        """
        only_sites = get_only_sites_from_structure(structure, migrating_specie)
        migration_graph = StructureGraph.with_local_env_strategy(
            only_sites,
            MinimumDistanceNN(cutoff=max_distance, get_all_sites=True),
        )
        return cls(structure=structure, m_graph=migration_graph, **kwargs)

    @staticmethod
    def get_structure_from_entries(
        entries: list[ComputedStructureEntry],
        migrating_ion_entry: ComputedEntry,
        **kwargs,
    ) -> list[Structure]:
        """
        Read in a list of base entries and inserted entries.  Return a list of structures that contains metastable
        sites for the migration species decorated with a "insertion_energy" property.

        Args:
            entries: list of entries, must contain a mixture of inserted and empty structures.
            migrating_ion_entry: The metallic phase of the working ion, used to calculate insertion energies.
            **kwargs: Passthrough for kwargs.

        Additional Kwargs:
            symprec:  symmetry parameter for SpacegroupAnalyzer
            ltol: Fractional length tolerance for StructureMatcher
            stol: Site tolerance for StructureMatcher
            angle_tol: Angle tolerance for StructureMatcher and SpacegroupAnalyzer
            only_single_cat: If True, only use single cation insertions so the
            site energy is more accurate use_strict_tol: halve the ltol and
            stol parameter for more strict matching.

        Returns:
            a list of host structures with working ion on all the metastable sites.
            The structures are ranked by the number of metastable sites (most is first)
            If the given entries are not enough to construct such a structure, return an empty list.

        """
        if len(migrating_ion_entry.composition.elements) != 1:
            raise RuntimeError("migrating_ion_entry should only have one element.")

        migration_element = migrating_ion_entry.composition.elements[0]

        base_entries = []
        inserted_entries = []
        for ient in entries:
            if migration_element in ient.composition.elements:
                inserted_entries.append(ient)
            else:
                base_entries.append(ient)

        if len(base_entries) == 0:
            logger.debug(
                f"No base entries found among {[ient.composition.formula for ient in entries]}, "
                "make sure you include one."
            )
            return []

        if len(inserted_entries) == 0:
            logger.debug(
                f"No inserted entries found among {[ient.composition.formula for ient in entries]}, "
                "make sure you include one."
            )
            return []

        l_base_and_inserted = process_entries(
            base_entries=base_entries,
            inserted_entries=inserted_entries,
            migrating_ion_entry=migrating_ion_entry,
            **kwargs,
        )
        res = []
        for group in l_base_and_inserted:
            all_sites = group["base"].copy().sites
            for isite in group["inserted"]:
                all_sites.append(isite)
            struct = Structure.from_sites(all_sites)
            # make spglib ignore all magmoms
            for isite in struct.sites:
                isite.properties.pop("magmom", None)
            res.append(struct)
        return res

    def _get_pos_and_migration_hop(self, u: int, v: int, w: int) -> None:
        """
        Insert a single MigrationHop object on a graph edge
        Args:
          u (int): index of initial node
          v (int): index of final node
          w (int): index for multiple edges that share the same two nodes.
        """
        edge = self.m_graph.graph[u][v][w]
        i_site = self.only_sites.sites[u]
        e_site = PeriodicSite(
            self.only_sites.sites[v].species,
            self.only_sites.sites[v].frac_coords + np.array(edge["to_jimage"]),
            lattice=self.only_sites.lattice,
        )
        # Positions might be useful for plotting
        edge["ipos"] = i_site.frac_coords
        edge["epos"] = e_site.frac_coords
        edge["ipos_cart"] = np.dot(i_site.frac_coords, self.only_sites.lattice.matrix)
        edge["epos_cart"] = np.dot(e_site.frac_coords, self.only_sites.lattice.matrix)

        edge["hop"] = MigrationHop(i_site, e_site, self.symm_structure, symprec=self.symprec)

    def _populate_edges_with_migration_hops(self) -> None:
        """Populate the edges with the data for the Migration Paths."""
        list(starmap(self._get_pos_and_migration_hop, self.m_graph.graph.edges))

    def _group_and_label_hops(self) -> dict:
        """Group the MigrationHop objects together and label all the symmetrically equlivaelnt hops with the same label."""
        hops = list(nx.get_edge_attributes(self.m_graph.graph, "hop").items())
        labs = generic_groupby(hops, comp=lambda x, y: x[1] == y[1])
        new_attr = {g_index: {"hop_label": labs[edge_index]} for edge_index, (g_index, _) in enumerate(hops)}
        nx.set_edge_attributes(self.m_graph.graph, new_attr)
        return new_attr

    def add_data_to_similar_edges(
        self,
        target_label: int | str,
        data: dict,
        m_hop: MigrationHop | None = None,
    ) -> None:
        """
        Insert data to all edges with the same label
        Args:
            target_label: The edge uniqueness label are adding data
            data: The data to passed to the different edges
            m_hop: If the data is an array, and m_hop is set, it uses the reference migration path to
            determine whether the data needs to be flipped so that 0-->1 is different from 1-->0.
        """
        for _u, _v, d in self.m_graph.graph.edges(data=True):
            if d["hop_label"] == target_label:
                d.update(data)
                # Try to override the data.
                if m_hop is not None and not m_hop.symm_structure.spacegroup.are_symmetrically_equivalent(
                    [m_hop.isite], [d["hop"].isite]
                ):
                    # "The data going to this edge needs to be flipped"
                    for k in data:  # noqa: PLC0206
                        if isinstance(data[k], np.ndarray | np.generic):
                            raise Warning("The data provided will only be flipped if it a list")
                        if not isinstance(data[k], list):
                            continue
                        d[k] = d[k][::-1]  # flip the data in the array

    def assign_cost_to_graph(self, cost_keys: list | None = None) -> None:
        """
        Read the data dict on each add and populate a cost key
        Args:
            cost_keys: a list of keys for data on each edge.
                The SC Graph is decorated with a "cost" key that is the product of the different keys here.
        """
        if cost_keys is None:
            cost_keys = ["hop_distance"]
        for k, v in self.unique_hops.items():
            cost_val = np.prod([v[ik] for ik in cost_keys])
            self.add_data_to_similar_edges(k, {"cost": cost_val})

    def get_path(self, max_val: float = 100000, flip_hops: bool = True) -> Generator:
        """
        Obtain a pathway through the material using hops that are in the current graph
        Basic idea:
            Get an endpoint p1 in the graph that is outside the current unit cell
            Ask the graph for a pathway that connects to p1 from either within the (0,0,0) cell
            or any other neighboring UC not containing p1.

        Args:
            max_val: Filter the graph by a cost
            flip_hops: If true, hops in paths returned will be flipped so
                isites and esites match to form a coherent path.
                If false, hops will retain their original orientation
                from the migration graph.

        Returns:
            Generator for list of Dicts:
            Each dict contains the information of a hop
        """
        if len(self.unique_hops) != len(self.unique_hops):
            logger.error(f"There are {len(self.unique_hops)} SC hops but {len(self.unique_hops)} UC hops in {self}")

        # for u, v, k, d in self.m_graph.graph.edges(data=True, keys=True):
        for u in self.m_graph.graph.nodes():
            # Create a copy of the graph so that we can trim the higher cost hops
            path_graph = deepcopy(self.m_graph.graph)
            # Trim the higher cost edges from the network
            cut_edges = []
            for tmp_u, tmp_v, tmp_k, tmp_d in path_graph.edges(data=True, keys=True):
                if tmp_d["cost"] > max_val:
                    cut_edges.append((tmp_u, tmp_v, tmp_k))
            for tmp_u, tmp_v, tmp_k in cut_edges:
                path_graph.remove_edge(tmp_u, tmp_v, key=tmp_k)
            # populate the entire graph with multiple images
            best_ans, path_parent = periodic_dijkstra(path_graph, sources={u}, weight="cost", max_image=2)
            # find a way to a u site that is not in the (0,0,0) image
            all_paths = []
            for idx, jimage in path_parent:
                if idx == u and jimage != (0, 0, 0):
                    path = [*get_optimal_pathway_rev(path_parent, (idx, jimage))][::-1]
                    assert path[-1][0] == u
                    all_paths.append(path)

            if len(all_paths) == 0:
                continue
            # The first hop must be one that leaves the 000 unit cell
            path = min(all_paths, key=lambda x: best_ans[x[-1]])

            # get the sequence of MigrationHop objects the represent the pathway
            path_hops = []
            for (idx1, jimage1), (idx2, jimage2) in itertools.pairwise(path):
                # for each pair of points in the periodic graph path look for end points in the original graph
                # the index pair has to be u->v with u <= v
                # once that is determined look up all such pairs in the graph and see if relative image
                # displacement +/- (jimage1 - jimage2) is present on of of the edges
                # Note: there should only ever be one valid to_jimage for a u->v pair
                i1_, i2_ = sorted((idx1, idx2))
                all_edge_data = [*path_graph.get_edge_data(i1_, i2_, default={}).items()]
                image_diff = np.subtract(jimage2, jimage1)
                found_ = 0
                for _k, tmp_d in all_edge_data:
                    if tmp_d["to_jimage"] in {tuple(image_diff), tuple(-image_diff)}:
                        path_hops.append(tmp_d)
                        found_ += 1
                if found_ != 1:
                    raise RuntimeError("More than one edge matched in original graph.")
            if flip_hops is True:  # flip hops in path to form coherent pathway
                yield u, order_path(path_hops, u)
            else:
                yield u, path_hops

    def get_summary_dict(self, added_keys: list[str] | None = None) -> dict:
        """
        Dictionary format, for saving to database.

        Args:
            added_keys: a list of keys for data on each edge.

        Returns:
            Dict.
        """
        hops = []
        keys = ["hop_label", "to_jimage", "ipos", "epos", "ipos_cart", "epos_cart"]

        if added_keys is not None:
            keys += added_keys

        def get_keys(d: dict) -> dict:
            return {k_: d[k_] for k_ in keys if k_ in d}

        for u, v, d in self.m_graph.graph.edges(data=True):
            new_hop = get_keys(d)
            new_hop["iindex"] = u
            new_hop["eindex"] = v
            hops.append(new_hop)

        unique_hops = []
        for d in self.unique_hops.values():
            new_hop["iindex"] = d["iindex"]  # type: ignore
            new_hop["eindex"] = d["eindex"]  # type: ignore
            unique_hops.append(get_keys(d))

        unique_hops = sorted(unique_hops, key=lambda x: x["hop_label"])

        return dict(
            structure=self.structure.as_dict(),
            host_structure=self.host_structure.as_dict(),
            migrating_specie=list(self.only_sites.composition.as_dict().keys()),
            hops=hops,
            unique_hops=unique_hops,
        )


class ChargeBarrierGraph(MigrationGraph):
    """A Migration graph with additional charge density analysis on the charge density of the host material."""

    def __init__(
        self,
        structure: Structure,
        m_graph: StructureGraph,
        potential_field: VolumetricData,
        potential_data_key: str,
        **kwargs,
    ) -> None:
        """
        Construct the MigrationGraph object using a VolumetricData object.
        The graph is constructed using the structure, and cost values are assigned based on charge density analysis.

        Args:
            structure (Structure): Input structure.
            m_graph (StructureGraph): Input structure graph.
            potential_field: Input VolumetricData object that describes the field does
                not have to contains all the metastable sites.
            potential_data_key (str): Key for potential data.
            **kwargs: Passthru for kwargs.
        """
        self.potential_field = potential_field
        self.potential_data_key = potential_data_key
        super().__init__(structure=structure, m_graph=m_graph, **kwargs)
        self._setup_grids()

    def _setup_grids(self) -> None:
        """Populate the internal variables used for defining the grid points in the charge density analysis."""
        # set up the grid
        aa = np.linspace(0, 1, len(self.potential_field.get_axis_grid(0)), endpoint=False)
        bb = np.linspace(0, 1, len(self.potential_field.get_axis_grid(1)), endpoint=False)
        cc = np.linspace(0, 1, len(self.potential_field.get_axis_grid(2)), endpoint=False)
        # move the grid points to the center
        aa, bb, dd = map(_shift_grid, [aa, bb, cc])

        # mesh grid for each unit cell
        AA, BB, CC = np.meshgrid(aa, bb, cc, indexing="ij")

        # should be using a mesh grid of 5x5x5 (using 3x3x3 misses some fringe cases)
        # but using 3x3x3 is much faster and only crops the cyliners in some rare case
        # if you keep the tube_radius small then this is not a big deal
        IMA, IMB, IMC = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")

        # store these
        self._uc_grid_shape = AA.shape
        self._fcoords = np.vstack([AA.flatten(), BB.flatten(), CC.flatten()]).T
        self._images = np.vstack([IMA.flatten(), IMB.flatten(), IMC.flatten()]).T

    def _dist_mat(self, pos_frac: np.ndarray) -> np.ndarray:
        # return a matrix that contains the distances to pos_frac
        aa = np.linspace(0, 1, len(self.potential_field.get_axis_grid(0)), endpoint=False)
        bb = np.linspace(0, 1, len(self.potential_field.get_axis_grid(1)), endpoint=False)
        cc = np.linspace(0, 1, len(self.potential_field.get_axis_grid(2)), endpoint=False)
        aa, bb, cc = map(_shift_grid, [aa, bb, cc])
        AA, BB, CC = np.meshgrid(aa, bb, cc, indexing="ij")
        dist_from_pos = self.potential_field.structure.lattice.get_all_distances(
            np.vstack([AA.flatten(), BB.flatten(), CC.flatten()]).T,
            pos_frac,
        )
        return dist_from_pos.reshape(AA.shape)

    def _get_pathfinder_from_hop(self, migration_hop: MigrationHop, n_images: int = 20) -> NEBPathfinder:
        # get migration pathfinder objects which contains the paths
        ipos = migration_hop.isite.frac_coords
        epos = migration_hop.esite.frac_coords
        mpos = migration_hop.esite.frac_coords

        start_struct = self.potential_field.structure.copy()
        end_struct = self.potential_field.structure.copy()
        mid_struct = self.potential_field.structure.copy()

        # the moving ion is always inserted on the zero index
        start_struct.insert(0, migration_hop.isite.species_string, ipos, properties=dict(magmom=0))
        end_struct.insert(0, migration_hop.isite.species_string, epos, properties=dict(magmom=0))
        mid_struct.insert(0, migration_hop.isite.species_string, mpos, properties=dict(magmom=0))

        chgpot = ChgcarPotential(self.potential_field, normalize=False)
        return NEBPathfinder(
            start_struct,
            end_struct,
            relax_sites=[0],
            v=chgpot.get_v(),
            n_images=n_images,
            mid_struct=mid_struct,
        )

    def _get_avg_chg_at_max(  # noqa: ANN202
        self,
        migration_hop: MigrationHop,
        radius: float | None = None,
        chg_along_path: bool = False,
        output_positions: bool = False,
    ):
        """Obtain the maximum average charge along the path
        Args:
            migration_hop (MigrationHop): MigrationPath object that represents a given hop
            radius (float, optional): radius of sphere to perform the average.
                    Defaults to None, which used the _tube_radius instead
            chg_along_path (bool, optional): If True, also return the entire list of average
                    charges along the path for plotting.
                    Defaults to False.
            output_positions (bool, optional): If True, also return the entire list of average
                    charges along the path for plotting.
                    Defaults to False.

        Returns:
            [float]: maximum of the charge density, (optional: entire list of charge density)
        """
        rr = radius or self._tube_radius
        if rr <= 0:  # type: ignore
            raise ValueError("The integration radius must be positive.")

        npf = self._get_pathfinder_from_hop(migration_hop)
        # get the charge in a sphere around each point
        centers = [image.sites[0].frac_coords for image in npf.images]
        avg_chg = []
        for ict in centers:
            dist_mat = self._dist_mat(ict)
            mask = dist_mat < rr
            vol_sphere = self.potential_field.structure.volume * (mask.sum() / self.potential_field.ngridpts)
            avg_chg.append(
                np.sum(self.potential_field.data[self.potential_data_key] * mask)
                / self.potential_field.ngridpts
                / vol_sphere
            )
        if output_positions:
            return max(avg_chg), avg_chg, centers
        if chg_along_path:
            return max(avg_chg), avg_chg
        return max(avg_chg)

    def _get_chg_between_sites_tube(self, migration_hop: MigrationHop, mask_file_seedname: str | None = None) -> float:
        """
        Calculate the amount of charge that a migrating ion has to move through in order to complete a hop
        Args:
            migration_hop: MigrationHop object that represents a given hop
            mask_file_seedname(string): seed name for output of the migration path masks (for debugging and
                visualization) (Default value = None).

        Returns:
            float: The total charge density in a tube that connects two sites of a given edges of the graph
        """
        try:
            _ = self._tube_radius
        except NameError:
            logger.warning("The radius of the tubes for charge analysis need to be defined first.")
        ipos = migration_hop.isite.frac_coords
        epos = migration_hop.esite.frac_coords

        cart_ipos = np.dot(ipos, self.potential_field.structure.lattice.matrix)
        cart_epos = np.dot(epos, self.potential_field.structure.lattice.matrix)
        pbc_mask = np.zeros(self._uc_grid_shape, dtype=bool).flatten()
        for img in self._images:
            grid_pos = np.dot(self._fcoords + img, self.potential_field.structure.lattice.matrix)
            proj_on_line = np.dot(grid_pos - cart_ipos, cart_epos - cart_ipos) / (np.linalg.norm(cart_epos - cart_ipos))
            dist_to_line = np.linalg.norm(
                np.cross(grid_pos - cart_ipos, cart_epos - cart_ipos) / (np.linalg.norm(cart_epos - cart_ipos)),
                axis=-1,
            )

            mask = (
                (proj_on_line >= 0)
                * (proj_on_line < np.linalg.norm(cart_epos - cart_ipos))
                * (dist_to_line < self._tube_radius)
            )
            pbc_mask = pbc_mask + mask
        pbc_mask = pbc_mask.reshape(self._uc_grid_shape)  # type: ignore

        if mask_file_seedname:
            mask_out = VolumetricData(
                structure=self.potential_field.structure.copy(),
                data={"total": self.potential_field.data["total"]},
            )
            mask_out.structure.insert(0, "X", ipos)
            mask_out.structure.insert(0, "X", epos)
            mask_out.data[self.potential_data_key] = pbc_mask
            isym = self.symm_structure.wyckoff_symbols[migration_hop.iindex]
            esym = self.symm_structure.wyckoff_symbols[migration_hop.eindex]
            mask_out.write_file(
                f"{mask_file_seedname}_{isym}_{esym}_tot({mask_out.data[self.potential_data_key].sum():.2f}).vasp"
            )

        return (
            self.potential_field.data[self.potential_data_key][pbc_mask].sum()
            / self.potential_field.ngridpts
            / self.potential_field.structure.volume
        )

    def populate_edges_with_chg_density_info(self, tube_radius: float = 1) -> None:
        """
        Args:
            tube_radius: Tube radius.
        """
        self._tube_radius = tube_radius
        for k, v in self.unique_hops.items():
            # charge in tube
            chg_tot = self._get_chg_between_sites_tube(v["hop"])
            self.add_data_to_similar_edges(k, {"chg_total": chg_tot})

            # max charge in sphere
            max_chg, avg_chg_list, frac_coords_list = self._get_avg_chg_at_max(
                v["hop"], chg_along_path=True, output_positions=True
            )
            images = [
                {"position": ifrac, "average_charge": ichg}
                for ifrac, ichg in zip(frac_coords_list, avg_chg_list, strict=False)
            ]
            v.update(
                dict(
                    chg_total=chg_tot,
                    max_avg_chg=max_chg,
                    images=images,
                )
            )
            self.add_data_to_similar_edges(k, {"max_avg_chg": max_chg})

    def get_least_chg_path(self) -> list:
        """
        Obtain an intercolating pathway through the material that has the least amount of charge
        Returns:
            list of hops.
        """
        min_chg = 100000000
        min_path = []
        all_paths = self.get_path()
        for path in all_paths:
            sum_chg = np.sum([hop[2]["chg_total"] for hop in path])
            sum_length = np.sum([hop[2]["hop"].length for hop in path])
            avg_chg = sum_chg / sum_length
            if avg_chg < min_chg:
                min_chg = sum_chg
                min_path = path
        return min_path

    def get_summary_dict(self, add_keys: list[str] | None = None) -> dict:
        """Dictionary format, for saving to database."""
        a_keys = ["max_avg_chg", "chg_total"]
        if add_keys is not None:
            a_keys += add_keys
        return super().get_summary_dict(added_keys=a_keys)


# Utility functions
def get_only_sites_from_structure(structure: Structure, migrating_specie: SpeciesLike) -> Structure:
    """
    Get a copy of the structure with only the migrating sites.

    Args:
        structure: The full_structure that contains all the sites
        migrating_specie: The name of migrating species
    Returns:
      Structure: Structure with all possible migrating ion sites
    """
    migrating_ion_sites = list(
        filter(
            lambda site: site.species == Composition({migrating_specie: 1}),
            structure.sites,
        )
    )
    return Structure.from_sites(migrating_ion_sites)


def _shift_grid(vv: np.ndarray) -> np.ndarray:
    """
    Move the grid points by half a step so that they sit in the center
    Args:
        vv: equally space grid points in 1-D.
    """
    step = vv[1] - vv[0]
    return vv + step / 2.0


def get_hop_site_sequence(hop_list: list[dict], start_u: int | str, key: str | None = None) -> list:
    """
    Read in a list of hop dictionaries and print the sequence of sites (and relevant property values if any).

    Args:
        hop_list: a list of the data on a sequence of hops
        start_u: the site index of the starting sites
        key (optional): property to track in a hop (e.g.: "hop_distance")

    Returns:
        String representation of the hop sequence (and property values if any)
    """
    hops = iter(hop_list)
    ihop = next(hops)
    site_seq = [ihop["eindex"], ihop["iindex"]] if ihop["eindex"] == start_u else [ihop["iindex"], ihop["eindex"]]

    for ihop in hops:
        if ihop["iindex"] == site_seq[-1]:
            site_seq.append(ihop["eindex"])
        elif ihop["eindex"] == site_seq[-1]:
            site_seq.append(ihop["iindex"])
        else:
            raise RuntimeError("The sequence of sites for the path is invalid.")

    if key is not None:
        key_seq = []
        hops = iter(hop_list)
        for ihop in hops:
            key_seq.append(ihop[key])
        return [site_seq, key_seq]

    return site_seq


def order_path(hop_list: list[dict], start_u: int | str) -> list[dict]:
    """
    Takes a list of hop dictionaries and flips hops (switches isite and esite)
    as needed to form a coherent path / sequence of sites according to
    get_hop_site_sequence().
    For example if hop_list = [{iindex:0, eindex:1, etc.}, {iindex:0, eindex:1, etc.}]
    then the output is [{iindex:0, eindex:1, etc.}, {iindex:1, eindex:0, etc.}] so that
    the following hop iindex matches the previous hop's eindex.

    Args:
        hop_list: a list of the data on a sequence of hops
        start_u: the site index of the starting sites
    Returns:
        a list of the data on a sequence of hops with hops in coherent orientation
    """
    seq = get_hop_site_sequence(hop_list, start_u)

    ordered_path = []
    for n, hop in zip(seq[:-1], hop_list, strict=False):
        if n == hop["iindex"]:  # don't flip hop
            ordered_path.append(hop)
        else:
            # create flipped hop
            fh = MigrationHop(
                isite=hop["hop"].esite,
                esite=hop["hop"].isite,
                symm_structure=hop["hop"].symm_structure,
                host_symm_struct=None,
                symprec=hop["hop"].symprec,
            )
            # must manually set iindex and eindex
            fh.iindex = hop["hop"].eindex
            fh.eindex = hop["hop"].iindex
            fhd = {
                "to_jimage": tuple(-1 * i for i in hop["to_jimage"]),
                "ipos": fh.isite.frac_coords,
                "epos": fh.esite.frac_coords,
                "ipos_cart": fh.isite.coords,
                "epos_cart": fh.esite.coords,
                "hop": fh,
                "hop_label": hop["hop_label"],
                "iindex": hop["eindex"],
                "eindex": hop["iindex"],
                "hop_distance": fh.length,
            }
            # flip any data that is in a list to match flipped hop orientation
            for k in hop:
                if k not in fhd:
                    if isinstance(hop[k], list):
                        fhd[k] = hop[k][::-1]
                    else:
                        fhd[k] = hop[k]
            ordered_path.append(fhd)

    return ordered_path


"""
Note the current pathway algorithm no longer needs supercells but the following
functions might still be useful for other applications
Finding all possible pathways in the periodic network is not possible.
We can do a good enough job if we make a (2x2x2) supercell of the structure and find
migration events using the following procedure:
- Look for a hop that leaves the SC like A->B (B on the outside)
- then at A look for a pathway to the image of B inside the SC
"""


# Utility Functions for comparing UC and SC hops


def almost(a, b) -> bool:  # noqa: ANN001
    """Return true if the values are almost equal."""
    SMALL_VAL = 1e-4
    try:
        return all(almost(i, j) for i, j in zip(list(a), list(b), strict=False))
    except BaseException:
        if isinstance(a, int | float) and isinstance(b, int | float):
            return abs(a - b) < SMALL_VAL
        raise NotImplementedError


def check_uc_hop(sc_hop: MigrationHop, uc_hop: MigrationHop) -> tuple | None:
    """
    See if hop in the 2X2X2 supercell and a unit cell hop
    are equivalent under lattice translation.

    Args:
        sc_hop: MigrationHop object form pymatgen-diffusion.
        uc_hop: MigrationHop object form pymatgen-diffusion.

    Return:
        image vector of length 3
        Is the UC hop flip of the SC hop
    """
    directions = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )

    sc_ipos = [icoord * 2 for icoord in sc_hop.isite.frac_coords]
    sc_epos = [icoord * 2 for icoord in sc_hop.esite.frac_coords]
    sc_mpos = [icoord * 2 for icoord in sc_hop.msite.frac_coords]

    uc_ipos = uc_hop.isite.frac_coords
    uc_epos = uc_hop.esite.frac_coords
    uc_mpos = uc_hop.msite.frac_coords

    for idir in directions:
        tmp_m = uc_mpos + idir
        if almost(tmp_m, sc_mpos):
            tmp_i = uc_ipos + idir
            tmp_e = uc_epos + idir
            if almost(tmp_i, sc_ipos) and almost(tmp_e, sc_epos):
                return idir, False
            if almost(tmp_e, sc_ipos) and almost(tmp_i, sc_epos):
                return idir, True
    return None


def map_hop_sc2uc(sc_hop: MigrationHop, mg: MigrationGraph) -> dict:
    """
    Map a given hop in the SC onto the UC.

    Args:
        sc_hop: MigrationHop object form pymatgen-diffusion.
        mg: MigrationGraph object from pymatgen-diffusion.

    Note:
        For now assume that the SC is exactly 2x2x2 of the UC.
        Can add in the parsing of different SC's later
        For a migration event in the SC from (0.45,0,0)-->(0.55,0,0)
        the UC hop might be (0.9,0,0)-->(0.1,0,0)[img:100]
        for the inverse of (0.1,0,0)-->(-0.1,0,0) the code needs to account for both those cases
    """
    for u, v, d in mg.m_graph.graph.edges(data=True):
        chk_res = check_uc_hop(sc_hop=sc_hop, uc_hop=d["hop"])
        if chk_res is not None:
            assert almost(d["hop"].length, sc_hop.length)
            return dict(
                uc_u=u,
                uc_v=v,
                hop=d["hop"],
                shift=chk_res[0],
                flip=chk_res[1],
                hop_label=d["hop_label"],
            )
    raise AssertionError("Looking for a SC hop without a matching UC hop")
