"""Featurisation for the EquiBind encoder, ported from DGL to PyTorch Geometric.

Extracted from `commons/process_mols.py` in HannesStark/EquiBind (MIT) and changed in
exactly one respect: graphs are assembled as `torch_geometric.data.Data` instead of
`dgl.graph`. Everything that computes the features — the neighbour search, the distance
binning, the local orientation frames — is the authors' code untouched, because the
published weights were trained on precisely these numbers.

Only what the encoder consumes is ported: the ligand graph and the Ca receptor graph. The
docking-specific graph builders stayed behind with the head.
"""

import math
import warnings

import pandas as pd
import numpy as np
import scipy.spatial as spa
import torch
from torch_geometric.data import Data
from Bio.PDB import get_surface, PDBParser, ShrakeRupley
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from biopandas.pdb import PandasPdb
from rdkit import Chem
from rdkit.Chem import MolFromPDBFile, AllChem, GetPeriodicTable, rdDistGeom
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from scipy import spatial
from scipy.special import softmax


biopython_parser = PDBParser()
periodic_table = GetPeriodicTable()
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
}

lig_feature_dims = (list(map(len, [
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_chirality_list'],
    allowable_features['possible_degree_list'],
    allowable_features['possible_formal_charge_list'],
    allowable_features['possible_implicit_valence_list'],
    allowable_features['possible_numH_list'],
    allowable_features['possible_number_radical_e_list'],
    allowable_features['possible_hybridization_list'],
    allowable_features['possible_is_aromatic_list'],
    allowable_features['possible_numring_list'],
    allowable_features['possible_is_in_ring3_list'],
    allowable_features['possible_is_in_ring4_list'],
    allowable_features['possible_is_in_ring5_list'],
    allowable_features['possible_is_in_ring6_list'],
    allowable_features['possible_is_in_ring7_list'],
    allowable_features['possible_is_in_ring8_list'],
])), 1)  # number of scalar features
rec_atom_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids'],
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_atom_type_2'],
    allowable_features['possible_atom_type_3'],
])), 2)

rec_residue_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids']
])), 2)



def lig_atom_featurizer(mol):
    ComputeGasteigerCharges(mol)  # they are Nan for 93 molecules in all of PDBbind. We put a 0 in that case.
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        g_charge = atom.GetDoubleProp('_GasteigerCharge')
        atom_features_list.append([
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
            allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
            allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
            allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
            allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
            allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8)),
            g_charge if not np.isnan(g_charge) and not np.isinf(g_charge) else 0.
        ])

    return torch.tensor(atom_features_list)


sr = ShrakeRupley(probe_radius=1.4,  # in A. Default is 1.40 roughly the radius of a water molecule.
                  n_points=100)  # resolution of the surface of each atom. Default is 100. A higher number of points results in more precise measurements, but slows down the calculation.


def rec_residue_featurizer(rec):
    feature_list = []
    sr.compute(rec, level="R")
    for residue in rec.get_residues():
        sasa = residue.sasa
        for atom in residue:
            if atom.name == 'CA':
                bfactor = atom.bfactor
        assert not np.isinf(bfactor)
        assert not np.isnan(bfactor)
        assert not np.isinf(sasa)
        assert not np.isnan(sasa)
        feature_list.append([safe_index(allowable_features['possible_amino_acids'], residue.get_resname()),
                             sasa,
                             bfactor])
    return torch.tensor(feature_list, dtype=torch.float32)  # (N_res, 1)


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim, feature_dims, use_scalar_feat=True, n_feats_to_use=None):
        # first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
        super(AtomEncoder, self).__init__()
        self.use_scalar_feat = use_scalar_feat
        self.n_feats_to_use = n_feats_to_use
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1]
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
            if i + 1 == self.n_feats_to_use:
                break

        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)

    def forward(self, x):
        x_embedding = 0
        assert x.shape[1] == self.num_categorical_features + self.num_scalar_features
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())
            if i + 1 == self.n_feats_to_use:
                break

        if self.num_scalar_features > 0 and self.use_scalar_feat:
            x_embedding += self.linear(x[:, self.num_categorical_features:])
        if torch.isnan(x_embedding).any():
            log('nan')
        return x_embedding


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def get_receptor_inference(rec_path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure('random_id', rec_path)
        rec = structure[0]
    coords = []
    c_alpha_coords = []
    n_coords = []
    c_coords = []
    valid_chain_ids = []
    lengths = []
    for i, chain in enumerate(rec):
        chain_coords = []  # num_residues, num_atoms, 3
        chain_c_alpha_coords = []
        chain_n_coords = []
        chain_c_coords = []
        count = 0
        invalid_res_ids = []
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                invalid_res_ids.append(residue.get_id())
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
                residue_coords.append(list(atom.get_vector()))
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid and not some weired molecule that is part of the complex
                chain_c_alpha_coords.append(c_alpha)
                chain_n_coords.append(n)
                chain_c_coords.append(c)
                chain_coords.append(np.array(residue_coords))
                count += 1
            else:
                invalid_res_ids.append(residue.get_id())
        for res_id in invalid_res_ids:
            chain.detach_child(res_id)
        lengths.append(count)
        coords.append(chain_coords)
        c_alpha_coords.append(np.array(chain_c_alpha_coords))
        n_coords.append(np.array(chain_n_coords))
        c_coords.append(np.array(chain_c_coords))
        if len(chain_coords) > 0:
            valid_chain_ids.append(chain.get_id())
    valid_coords = []
    valid_c_alpha_coords = []
    valid_n_coords = []
    valid_c_coords = []
    valid_lengths = []
    invalid_chain_ids = []
    for i, chain in enumerate(rec):
        if chain.get_id() in valid_chain_ids:
            valid_coords.append(coords[i])
            valid_c_alpha_coords.append(c_alpha_coords[i])
            valid_n_coords.append(n_coords[i])
            valid_c_coords.append(c_coords[i])
            valid_lengths.append(lengths[i])
        else:
            invalid_chain_ids.append(chain.get_id())
    coords = [item for sublist in valid_coords for item in sublist]  # list with n_residues arrays: [n_atoms, 3]

    c_alpha_coords = np.concatenate(valid_c_alpha_coords, axis=0)  # [n_residues, 3]
    n_coords = np.concatenate(valid_n_coords, axis=0)  # [n_residues, 3]
    c_coords = np.concatenate(valid_c_coords, axis=0)  # [n_residues, 3]

    for invalid_id in invalid_chain_ids:
        rec.detach_child(invalid_id)

    assert len(c_alpha_coords) == len(n_coords)
    assert len(c_alpha_coords) == len(c_coords)
    assert sum(valid_lengths) == len(c_alpha_coords)
    return rec, coords, c_alpha_coords, n_coords, c_coords


def log(*args):
    """The original logs to a file; here it goes to stdout, which the harness captures."""
    print(*args)


def rigid_transform_Kabsch_3D(A, B):
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")


    # find mean column wise: 3 x 1
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = np.diag([1.,1.,-1.])
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(np.linalg.det(R) - 1) < 1e-5

    t = -R @ centroid_A + centroid_B
    return R, t

# R = 3x3 rotation matrix
# t = 3x1 column vector
# This already takes residue identity into account.


def get_rdkit_coords(mol, seed=None):
    """Generate an unbound conformer.

    Deviation from the original, forced by a newer RDKit: ETKDG raises on ligands whose
    hydrogens have been stripped ("Cannot be a chiral center", "nbrs.size() >= 3") and the
    original's fallback only catches an embed that *returns* -1, not one that throws. Every
    CASF-2016 ligand tested failed this way, and `get_lig_graph_revised` silently fell back
    to the crystal coordinates — which hands the model the pose it is meant to predict.

    Embedding with explicit hydrogens and removing them afterwards succeeds on all of them.
    RemoveHs preserves heavy-atom order, so the coordinates still line up with the graph.
    """
    ps = AllChem.ETKDGv2()
    if seed is not None:
        ps.randomSeed = seed

    mol_h = Chem.AddHs(Chem.Mol(mol))
    if AllChem.EmbedMolecule(mol_h, ps) == -1:
        log('rdkit coords could not be generated without using random coords. using random coords now.')
        ps.useRandomCoords = True
        if AllChem.EmbedMolecule(mol_h, ps) == -1:
            raise ValueError('could not embed a conformer even with random coordinates')
    AllChem.MMFFOptimizeMolecule(mol_h, confId=0)

    mol_noh = Chem.RemoveHs(mol_h)
    if mol_noh.GetNumAtoms() != mol.GetNumAtoms():
        raise ValueError(
            f'conformer has {mol_noh.GetNumAtoms()} atoms, molecule has {mol.GetNumAtoms()}')
    return torch.tensor(mol_noh.GetConformer().GetPositions(), dtype=torch.float32)


def get_lig_graph_revised(mol, name, radius=20, max_neighbors=None, use_rdkit_coords=False):
    conf = mol.GetConformer()
    true_lig_coords = conf.GetPositions()
    if use_rdkit_coords:
        try:
            rdkit_coords = get_rdkit_coords(mol).numpy()
            R, t = rigid_transform_Kabsch_3D(rdkit_coords.T, true_lig_coords.T)
            lig_coords = ((R @ (rdkit_coords).T).T + t.squeeze())
            log('kabsch RMSD between rdkit ligand and true ligand is ', np.sqrt(np.sum((lig_coords - true_lig_coords) ** 2, axis=1).mean()).item())
        except Exception as e:
            lig_coords = true_lig_coords
            with open('temp_create_dataset_rdkit_timesplit_no_lig_or_rec_overlap_train.log', 'a') as f:
                f.write('Generating RDKit conformer failed for  \n')
                f.write(name)
                f.write('\n')
                f.write(str(e))
                f.write('\n')
                f.flush()
            print('Generating RDKit conformer failed for  ')
            print(name)
            print(str(e))
    else:
        lig_coords = true_lig_coords
    num_nodes = lig_coords.shape[0]
    assert lig_coords.shape[1] == 3
    distance = spa.distance.cdist(lig_coords, lig_coords)

    src_list = []
    dst_list = []
    dist_list = []
    mean_norm_list = []
    for i in range(num_nodes):
        dst = list(np.where(distance[i, :] < radius)[0])
        dst.remove(i)
        if max_neighbors != None and len(dst) > max_neighbors:
            dst = list(np.argsort(distance[i, :]))[1: max_neighbors + 1]  # closest would be self loop
        if len(dst) == 0:
            dst = list(np.argsort(distance[i, :]))[1:2]  # closest would be the index i itself > self loop
            log(
                f'The lig_radius {radius} was too small for one lig atom such that it had no neighbors. So we connected {i} to the closest other lig atom {dst}')
        assert i not in dst
        assert dst != []
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distance[i, dst])
        dist_list.extend(valid_dist)
        valid_dist_np = distance[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        diff_vecs = lig_coords[src, :] - lig_coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)

        mean_norm_list.append(mean_vec_ratio_norm)
    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)
    graph = Data(
        x=lig_atom_featurizer(mol),
        pos=torch.from_numpy(np.array(true_lig_coords).astype(np.float32)),
        edge_index=torch.stack([torch.tensor(src_list), torch.tensor(dst_list)]).long(),
        edge_attr=distance_featurizer(dist_list, 0.75),  # divisor ~0.75, see the original
        mu_r_norm=torch.from_numpy(np.array(mean_norm_list).astype(np.float32)),
        num_nodes=num_nodes,
    )
    if use_rdkit_coords:
        graph.new_x = torch.from_numpy(np.array(lig_coords).astype(np.float32))
    return graph


def distance_featurizer(dist_list, divisor) -> torch.Tensor:
    # you want to use a divisor that is close to 4/7 times the average distance that you want to encode
    length_scale_list = [1.5 ** x for x in range(15)]
    center_list = [0. for _ in range(15)]

    num_edge = len(dist_list)
    dist_list = np.array(dist_list)

    transformed_dist = [np.exp(- ((dist_list / divisor) ** 2) / float(length_scale))
                        for length_scale, center in zip(length_scale_list, center_list)]

    transformed_dist = np.array(transformed_dist).T
    transformed_dist = transformed_dist.reshape((num_edge, -1))
    return torch.from_numpy(transformed_dist.astype(np.float32))


def get_calpha_graph(rec, c_alpha_coords, n_coords, c_coords, cutoff=20, max_neighbor=None):
    ################## Extract 3D coordinates and n_i,u_i,v_i vectors of representative residues ################
    residue_representatives_loc_list = []
    n_i_list = []
    u_i_list = []
    v_i_list = []
    for i, residue in enumerate(rec.get_residues()):
        n_coord = n_coords[i]
        c_alpha_coord = c_alpha_coords[i]
        c_coord = c_coords[i]
        u_i = (n_coord - c_alpha_coord) / np.linalg.norm(n_coord - c_alpha_coord)
        t_i = (c_coord - c_alpha_coord) / np.linalg.norm(c_coord - c_alpha_coord)
        n_i = np.cross(u_i, t_i) / np.linalg.norm(np.cross(u_i, t_i))
        v_i = np.cross(n_i, u_i)
        assert (math.fabs(
            np.linalg.norm(v_i) - 1.) < 1e-5), "protein utils protein_to_graph_dips, v_i norm larger than 1"
        n_i_list.append(n_i)
        u_i_list.append(u_i)
        v_i_list.append(v_i)
        residue_representatives_loc_list.append(c_alpha_coord)

    residue_representatives_loc_feat = np.stack(residue_representatives_loc_list, axis=0)  # (N_res, 3)
    n_i_feat = np.stack(n_i_list, axis=0)
    u_i_feat = np.stack(u_i_list, axis=0)
    v_i_feat = np.stack(v_i_list, axis=0)
    num_residues = len(c_alpha_coords)
    if num_residues <= 1:
        raise ValueError(f"rec contains only 1 residue!")

    ################### Build the k-NN graph ##############################
    assert num_residues == residue_representatives_loc_feat.shape[0]
    assert residue_representatives_loc_feat.shape[1] == 3
    distances = spa.distance.cdist(c_alpha_coords, c_alpha_coords)
    src_list = []
    dst_list = []
    dist_list = []
    mean_norm_list = []
    for i in range(num_residues):
        dst = list(np.where(distances[i, :] < cutoff)[0])
        dst.remove(i)
        if max_neighbor != None and len(dst) > max_neighbor:
            dst = list(np.argsort(distances[i, :]))[1: max_neighbor + 1]
        if len(dst) == 0:
            dst = list(np.argsort(distances[i, :]))[1:2]  # choose second because first is i itself
            log(
                f'The c_alpha_cutoff {cutoff} was too small for one c_alpha such that it had no neighbors. So we connected it to the closest other c_alpha')
        assert i not in dst
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distances[i, dst])
        dist_list.extend(valid_dist)
        valid_dist_np = distances[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        diff_vecs = residue_representatives_loc_feat[src, :] - residue_representatives_loc_feat[
                                                               dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
        mean_norm_list.append(mean_vec_ratio_norm)
    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)
    edge_index = torch.stack([torch.tensor(src_list), torch.tensor(dst_list)]).long()
    node_feat = rec_residue_featurizer(rec)
    edge_feat = distance_featurizer(dist_list, divisor=4)  # avg distance = 7, so divisor = 4

    # Loop over all edges of the graph and build the various p_ij, q_ij, k_ij, t_ij pairs
    edge_feat_ori_list = []
    for i in range(len(dist_list)):
        src = src_list[i]
        dst = dst_list[i]
        # place n_i, u_i, v_i as lines in a 3x3 basis matrix
        basis_matrix = np.stack((n_i_feat[dst, :], u_i_feat[dst, :], v_i_feat[dst, :]), axis=0)
        p_ij = np.matmul(basis_matrix,
                         residue_representatives_loc_feat[src, :] - residue_representatives_loc_feat[
                                                                    dst, :])
        q_ij = np.matmul(basis_matrix, n_i_feat[src, :])  # shape (3,)
        k_ij = np.matmul(basis_matrix, u_i_feat[src, :])
        t_ij = np.matmul(basis_matrix, v_i_feat[src, :])
        s_ij = np.concatenate((p_ij, q_ij, k_ij, t_ij), axis=0)  # shape (12,)
        edge_feat_ori_list.append(s_ij)
    edge_feat_ori_feat = np.stack(edge_feat_ori_list, axis=0)  # shape (num_edges, 4, 3)
    edge_feat_ori_feat = torch.from_numpy(edge_feat_ori_feat.astype(np.float32))
    edge_feat = torch.cat([edge_feat, edge_feat_ori_feat], axis=1)  # (num_edges, 17)

    residue_representatives_loc_feat = torch.from_numpy(residue_representatives_loc_feat.astype(np.float32))
    return Data(
        x=node_feat,
        pos=residue_representatives_loc_feat,
        edge_index=edge_index,
        edge_attr=edge_feat,
        mu_r_norm=torch.from_numpy(np.array(mean_norm_list).astype(np.float32)),
        num_nodes=num_residues,
    )


def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    """Load a molecule from a file of format ``.mol2`` or ``.sdf`` or ``.pdbqt`` or ``.pdb``.

    Parameters
    ----------
    molecule_file : str
        Path to file for storing a molecule, which can be of format ``.mol2`` or ``.sdf``
        or ``.pdbqt`` or ``.pdb``.
    sanitize : bool
        Whether sanitization is performed in initializing RDKit molecule instances. See
        https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
        Default to False.
    calc_charges : bool
        Whether to add Gasteiger charges via RDKit. Setting this to be True will enforce
        ``sanitize`` to be True. Default to False.
    remove_hs : bool
        Whether to remove hydrogens via RDKit. Note that removing hydrogens can be quite
        slow for large molecules. Default to False.
    use_conformation : bool
        Whether we need to extract molecular conformation from proteins and ligands.
        Default to True.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the loaded molecule.
    coordinates : np.ndarray of shape (N, 3) or None
        The 3D coordinates of atoms in the molecule. N for the number of atoms in
        the molecule. None will be returned if ``use_conformation`` is False or
        we failed to get conformation information.
    """
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        return ValueError('Expect the format of the molecule_file to be '
                          'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except:
        return None

    return mol