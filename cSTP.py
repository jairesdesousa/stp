#!/usr/bin/env python3
"""
Calculate Signed Scalar Triple Vectors from SDF files based on chiral centers.

Extended from: https://doi.org/10.1371/journal.pone.0333635

Atomic properties used:
- Mass (using weighted average from Equation 4)
- Gasteiger charge
- Absolute Gasteiger charge
- MMFF charge
- Absolute MMFF charge
- Polarizability (atomic)
- Crippen logP contribution
- Crippen MR contribution
- TPSA contribution
- Labute ASA contribution
- EState index

Features:
- Centers calculations on chiral atoms
- Can restrict to atoms within specified bond distance from chiral center
- Sums triple vectors for all chiral atoms

Adapted from code written by generative artificial intelligence models DeepSeek
and ChatGPT in February 2026
"""

import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen
from rdkit.Chem import rdPartialCharges
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.EState import EStateIndices
from rdkit.Chem.rdchem import ChiralType
from rdkit.Chem import rdCIPLabeler
from collections import OrderedDict
import pandas as pd

# Atomic polarizabilities (in Å³) from static dipole polarizabilities of neutral elements
ATOMIC_POLARIZABILITIES = {
    1: 0.6668,    # H
    6: 1.760,     # C
    7: 1.100,     # N
    8: 0.802,     # O
    9: 0.557,     # F
    15: 3.630,    # P
    16: 2.900,    # S
    17: 2.180,    # Cl
    35: 3.050,    # Br
    53: 4.440,    # I
    'default': 2.000
}

def get_atomic_polarizability(atomic_num):
    """Get atomic polarizability for given atomic number"""
    return ATOMIC_POLARIZABILITIES.get(atomic_num, ATOMIC_POLARIZABILITIES['default'])

def get_chiral_atoms(mol):
    """Identify all chiral atoms in the molecule"""
    chiral_atoms = []
    mol_copy = Chem.Mol(mol)
    rdCIPLabeler.AssignCIPLabels(mol_copy)
    for atom in mol_copy.GetAtoms():
        if atom.HasProp('_CIPCode'):
            chiral_atoms.append(atom.GetIdx())
    return chiral_atoms

def get_atoms_within_distance(mol, center_atom_idx, max_distance):
    """
    Get atom indices within specified bond distance from center atom.
    
    Args:
        mol: RDKit molecule
        center_atom_idx: Index of chiral center atom
        max_distance: Maximum bond distance (None for all atoms)
    
    Returns:
        List of atom indices within the specified distance
    """
    if max_distance is None:
        # Return all atoms if no distance restriction
        return list(range(mol.GetNumAtoms()))
    
    # Calculate shortest path distances from chiral center
    distances = {}
    visited = set()
    queue = [(center_atom_idx, 0)]
    
    while queue:
        atom_idx, dist = queue.pop(0)
        if atom_idx in visited:
            continue
        visited.add(atom_idx)
        distances[atom_idx] = dist
        
        if dist < max_distance:
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx not in visited:
                    if neighbor.GetSymbol() == 'H' and dist > 0:
                        continue
                    queue.append((neighbor_idx, dist + 1))
    return list(distances.keys())

def center_on_atom(coords, center_atom_idx):
    """
    Translate coordinates so that the specified atom is at origin.
    
    Args:
        coords: Numpy array of coordinates
        center_atom_idx: Index of atom to center on
    
    Returns:
        Centered coordinates
    """
    coords = np.array(coords)
    center_coord = coords[center_atom_idx]
    centered_coords = coords - center_coord
    return centered_coords

def calculate_standard_property_vector(centered_coords, property_values, atom_indices=None):
    """
    Calculate property vector for non-mass properties (Equation 3 in paper)
    M_p = Σ_i p_i * r_i
    
    Args:
        centered_coords: Coordinates centered on chiral atom
        property_values: List of property values
        atom_indices: List of atom indices to include (None for all)
    """
    property_values = np.array(property_values)
    centered_coords = np.array(centered_coords)
    
    norms = np.linalg.norm(centered_coords, axis=1)
    norms_safe = np.where(norms == 0, 1, norms)    
    
    if atom_indices is not None:
        property_values = property_values[atom_indices]
        centered_coords = centered_coords[atom_indices]
        norms_safe = norms_safe[atom_indices]
    
    # Calculate the property vector: sum of property * coordinate
    M_p = np.sum(property_values.reshape(-1, 1) * centered_coords, axis=0)

    return M_p

def calculate_scalar_triple_product(M1, M2, M3):
    """
    Calculate SIGNED scalar triple product for three property vectors
    STP = (M1 × M2) · M3
    """
    # Calculate scalar triple product: (M1 × M2) · M3
    cross_prod = np.cross(M1, M2)
    stp = np.dot(cross_prod, M3)
    denomin_stp = pow(np.linalg.norm(M1)*np.linalg.norm(M2)*np.linalg.norm(M3), 2/3)
    if denomin_stp > 0:
        stp_norm = stp/denomin_stp
    else:
        stp_norm = stp
    
    if np.isnan(stp_norm):
        return 0
    else:
        return stp_norm

def get_crippen_contributions(mol):
    """
    Calculate Crippen logP and MR contributions using the proper public API
    """
    try:
        # Use the public functions to calculate Crippen contributions with proper parameters
        contribs = rdMolDescriptors._CalcCrippenContribs(mol, False, [], [])
        
        # Extract logP and MR contributions from the tuples
        # Each tuple is (logP_contrib, MR_contrib) for each atom
        logP_values = [contrib[0] for contrib in contribs]
        MR_values = [contrib[1] for contrib in contribs]
        
        return logP_values, MR_values
        
    except Exception as e:
        # Fallback: if the internal function fails, use simple approximations
        print(f"Warning: Using approximation for Crippen contributions: {str(e)}")
        n_atoms = mol.GetNumAtoms()
        
        # Simple approximations based on atom types
        logP_values = []
        MR_values = []
        
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            # Simple approximations based on common atom types
            if atomic_num == 1:  # H
                logP_contrib = 0.0
                MR_contrib = 0.1
            elif atomic_num == 6:  # C
                logP_contrib = 0.0
                MR_contrib = 0.2
            elif atomic_num == 7:  # N
                logP_contrib = -0.5
                MR_contrib = 0.3
            elif atomic_num == 8:  # O
                logP_contrib = -0.7
                MR_contrib = 0.4
            elif atomic_num == 9:  # F
                logP_contrib = -0.4
                MR_contrib = 0.3
            elif atomic_num == 17:  # Cl
                logP_contrib = 0.3
                MR_contrib = 0.8
            elif atomic_num == 35:  # Br
                logP_contrib = 0.5
                MR_contrib = 1.2
            elif atomic_num == 53:  # I
                logP_contrib = 0.7
                MR_contrib = 1.8
            else:
                logP_contrib = 0.0
                MR_contrib = 0.5
            
            logP_values.append(logP_contrib)
            MR_values.append(MR_contrib)
        
        return logP_values, MR_values

def get_tpsa_contributions(mol):
    """
    Calculate TPSA (Topological Polar Surface Area) atomic contributions
    """
    try:
        # rdMolDescriptors._CalcTPSAContribs returns atomic contributions to TPSA
        tpsa_contribs = rdMolDescriptors._CalcTPSAContribs(mol)
        return tpsa_contribs
    except Exception as e:
        print(f"Warning: Using approximation for TPSA contributions: {str(e)}")
        # Fallback approximation
        tpsa_values = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if atomic_num == 7 or atomic_num == 8:  # N or O
                if atom.GetHybridization() == Chem.HybridizationType.SP3:
                    tpsa_contrib = 3.0
                elif atom.GetHybridization() == Chem.HybridizationType.SP2:
                    tpsa_contrib = 2.5
                else:
                    tpsa_contrib = 2.0
            elif atomic_num == 16:  # S
                tpsa_contrib = 1.0
            elif atomic_num == 15:  # P
                tpsa_contrib = 3.5
            elif atomic_num == 9:  # F
                tpsa_contrib = 0.5
            else:
                tpsa_contrib = 0.0
            tpsa_values.append(tpsa_contrib)
        return tpsa_values

def get_labute_asa_contributions(mol):
    """
    Calculate Labute ASA (Approximate Surface Area) atomic contributions
    """
    try:
        # rdMolDescriptors._CalcLabuteASAContribs returns a tuple where first element is atomic contributions
        labute_result = rdMolDescriptors._CalcLabuteASAContribs(mol)
        asa_contribs = labute_result[0]  # First element contains atomic contributions
        return asa_contribs
    except Exception as e:
        print(f"Warning: Using approximation for Labute ASA contributions: {str(e)}")
        # Fallback approximation
        asa_contribs = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            # Rough approximation based on atomic van der Waals radii
            vdw_radii = {
                1: 1.20,   # H
                6: 1.70,   # C
                7: 1.55,   # N
                8: 1.52,   # O
                9: 1.47,   # F
                15: 1.80,  # P
                16: 1.80,  # S
                17: 1.75,  # Cl
                35: 1.85,  # Br
                53: 1.98   # I
            }
            radius = vdw_radii.get(atomic_num, 1.70)
            # Surface area is proportional to r^2
            asa_contrib = 4 * 3.14159 * (radius ** 2)
            asa_contribs.append(asa_contrib)
        
        # Normalize to match typical molecular ASA
        total_asa = sum(asa_contribs)
        if total_asa > 0:
            # Typical ASA for drug-like molecules is 200-600 Å²
            target_asa = 400.0
            asa_contribs = [contrib * target_asa / total_asa for contrib in asa_contribs]
        
        return asa_contribs

def get_estate_contributions(mol):
    """
    Calculate E-State atomic contributions using RDKit's proper implementation
    """
    try:
        estate_indices = EStateIndices(mol)
        return list(estate_indices)
    except Exception as e:
        print(f"Warning: Using approximation for E-State contributions: {str(e)}")
        # Fallback approximation based on atom connectivity
        estate_values = []
        for atom in mol.GetAtoms():
            # Simple approximation: more connected atoms have higher E-State
            degree = atom.GetDegree()
            atomic_num = atom.GetAtomicNum()
            base_value = 0.0
            if atomic_num == 6:  # C
                base_value = 1.0
            elif atomic_num == 7:  # N
                base_value = 2.0
            elif atomic_num == 8:  # O
                base_value = 3.0
            elif atomic_num == 9:  # F
                base_value = 4.0
            elif atomic_num == 1:  # H
                base_value = 0.1
            else:
                base_value = 1.5
            
            estate_value = base_value * (1 + 0.1 * degree)
            estate_values.append(estate_value)
        return estate_values

def get_mmff_charges(mol):
    """
    Calculate MMFF94 partial charges for all atoms using original coordinates
    Assumes the molecule already contains all hydrogens
    """
    try:
        from rdkit.Chem import AllChem
        
        # Use the original molecule with existing hydrogens and coordinates
        if AllChem.MMFFSanitizeMolecule(mol) == 0:  # 0 means success
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
            if mmff_props:
                charges = [mmff_props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())]
                return charges
        
        # If MMFF fails, fall back to Gasteiger charges
        print("Warning: MMFF94 charge calculation failed, using Gasteiger charges as fallback")
        return get_gasteiger_charges(mol)
        
    except Exception as e:
        print(f"Warning: MMFF94 charge calculation failed: {str(e)}")
        return get_gasteiger_charges(mol)

def get_gasteiger_charges(mol):
    """
    Calculate Gasteiger partial charges
    """
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            charge = 0.0
            if atom.HasProp("_GasteigerCharge"):
                try:
                    charge = float(atom.GetProp("_GasteigerCharge"))
                except:
                    charge = 0.0
            charges.append(charge)
        return charges
    except:
        return [0.0] * mol.GetNumAtoms()

def calculate_triple_vectors_for_chiral_center(mol, chiral_atom_idx, max_bond_distance=None):
    """
    Calculate signed scalar triple vectors for a specific chiral center.
    
    Args:
        mol: RDKit molecule
        chiral_atom_idx: Index of the chiral atom
        max_bond_distance: Maximum bond distance from chiral center (None for all atoms)
    
    Returns:
        Dictionary of triple vector values for this chiral center
    """
    # Check if molecule has 3D coordinates
    if not mol.GetNumConformers():
        raise ValueError("Molecule has no 3D coordinates")

    conf = mol.GetConformer()
    
    # Get atoms within specified distance from chiral center
    atom_indices = get_atoms_within_distance(mol, chiral_atom_idx, max_bond_distance)
    
    # Get atomic properties and original coordinates
    atomic_nums = []
    masses = []
    gasteiger_charges = []
    abs_gasteiger_charges = []
    polarizabilities = []
    original_coords = []
    
    # Calculate MMFF charges
    mmff_charges = get_mmff_charges(mol)
    mmff_charges = [c / 1.6217 for c in mmff_charges]
    abs_mmff_charges = [abs(charge) for charge in mmff_charges]
    
    # Calculate Gasteiger charges
    gasteiger_charges = get_gasteiger_charges(mol)
    gasteiger_charges = [c / 0.57267 for c in gasteiger_charges]
    abs_gasteiger_charges = [abs(charge) for charge in gasteiger_charges]
    
    # Get Crippen contributions
    logP_values, MR_values = get_crippen_contributions(mol)
    logP_values = [c / 1.027 for c in logP_values]
    MR_values = [c / 8.927 for c in MR_values]
    
    # Get TPSA contributions
    tpsa_values = get_tpsa_contributions(mol)
    tpsa_values = [c / 26.02 for c in tpsa_values]
    
    # Get Labute ASA contributions
    labute_asa_values = get_labute_asa_contributions(mol)
    labute_asa_values = [c / 15.92994 for c in labute_asa_values]
    
    # Get EState indices
    estate_values = get_estate_contributions(mol)
    estate_values = [c / 16.0961 for c in estate_values]

    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        atomic_num = atom.GetAtomicNum()
        atomic_nums.append(atomic_num)
        masses.append(atom.GetMass())
        polarizabilities.append(get_atomic_polarizability(atomic_num))
        
        # Get coordinates
        pos = conf.GetAtomPosition(i)
        original_coords.append([pos.x, pos.y, pos.z])
    
    masses = [c / 12 for c in masses]
    polarizabilities = [c / 3.05 for c in polarizabilities]
    
    # Convert to numpy arrays
    atomic_nums = np.array(atomic_nums)
    masses = np.array(masses)
    gasteiger_charges = np.array(gasteiger_charges)
    abs_gasteiger_charges = np.array(abs_gasteiger_charges)
    mmff_charges = np.array(mmff_charges)
    abs_mmff_charges = np.array(abs_mmff_charges)
    polarizabilities = np.array(polarizabilities)
    logP_values = np.array(logP_values)
    MR_values = np.array(MR_values)
    tpsa_values = np.array(tpsa_values)
    labute_asa_values = np.array(labute_asa_values)
    estate_values = np.array(estate_values)
    original_coords = np.array(original_coords)
    
    # Step 1: Center coordinates on chiral atom
    centered_coords = center_on_atom(original_coords, chiral_atom_idx)
    
    # Calculate property moments for the subset of atoms using Equation 3
    property_moments = {}

    property_moments['mass'] = calculate_standard_property_vector(centered_coords, masses, atom_indices)
    property_moments['gasteiger_charge'] = calculate_standard_property_vector(centered_coords, gasteiger_charges, atom_indices)
    property_moments['abs_gasteiger_charge'] = calculate_standard_property_vector(centered_coords, abs_gasteiger_charges, atom_indices)
    property_moments['mmff_charge'] = calculate_standard_property_vector(centered_coords, mmff_charges, atom_indices)
    property_moments['abs_mmff_charge'] = calculate_standard_property_vector(centered_coords, abs_mmff_charges, atom_indices)
    property_moments['polarizability'] = calculate_standard_property_vector(centered_coords, polarizabilities, atom_indices)
    property_moments['crippen_logP'] = calculate_standard_property_vector(centered_coords, logP_values, atom_indices)
    property_moments['crippen_MR'] = calculate_standard_property_vector(centered_coords, MR_values, atom_indices)
    property_moments['tpsa'] = calculate_standard_property_vector(centered_coords, tpsa_values, atom_indices)
    property_moments['labute_asa'] = calculate_standard_property_vector(centered_coords, labute_asa_values, atom_indices)
    property_moments['estate'] = calculate_standard_property_vector(centered_coords, estate_values, atom_indices)
    
    property_names = list(property_moments.keys())
    n_properties = len(property_names)
            
    for i in range(n_properties):
        if np.isnan(property_moments[property_names[i]][1]):
            print(f"No moment with {property_names[i]}!")
    
    # Calculate all unique triple combinations for this chiral center
    triple_vectors = OrderedDict()

    for i in range(n_properties):
        for j in range(i + 1, n_properties):
            for k in range(j + 1, n_properties):
                prop1 = property_names[i]
                prop2 = property_names[j]
                prop3 = property_names[k]
                
                key = f"STP_{prop1}_{prop2}_{prop3}"
                stp_value = calculate_scalar_triple_product(
                    property_moments[prop1],
                    property_moments[prop2],
                    property_moments[prop3]
                )
                triple_vectors[key] = stp_value

    return triple_vectors

def calculate_all_triple_vectors(mol, max_bond_distance=None):
    """
    Calculate all signed scalar triple vectors for a molecule, summing over all chiral centers.
    
    Args:
        mol: RDKit molecule
        max_bond_distance: Maximum bond distance from chiral centers (None for all atoms)
    
    Returns:
        Dictionary of summed triple vector values across all chiral centers
    """    
    # Initialize result dictionary with zeros
    property_names = ['mass', 'gasteiger_charge', 'abs_gasteiger_charge', 'mmff_charge', 
                      'abs_mmff_charge', 'polarizability', 'crippen_logP', 'crippen_MR', 
                      'tpsa', 'labute_asa', 'estate']
    
    # Generate all unique triple combinations
    triple_vectors_sum = OrderedDict()
    n_properties = len(property_names)
    
    for i in range(n_properties):
        for j in range(i + 1, n_properties):
            for k in range(j + 1, n_properties):
                prop1 = property_names[i]
                prop2 = property_names[j]
                prop3 = property_names[k]
                key = f"STP_{prop1}_{prop2}_{prop3}"
                triple_vectors_sum[key] = 0.0

    # Identify chiral atoms
    chiral_atoms = get_chiral_atoms(mol)
    
    if not chiral_atoms:
        print("Warning: No chiral atoms found in molecule.")
        # Fall back to original behavior using center of mass
        return triple_vectors_sum

    # Calculate triple vectors for each chiral center and sum them
    for chiral_idx in chiral_atoms:
        try:	
            triple_vectors = calculate_triple_vectors_for_chiral_center(mol, chiral_idx, max_bond_distance)
            # Add to sum
            for key in triple_vectors_sum:
                if key in triple_vectors:
                    triple_vectors_sum[key] += triple_vectors[key]
        except Exception as e:
            print(f"Warning: Failed to calculate triple vectors for chiral atom {chiral_idx}: {str(e)}")
            continue
    
    # Add metadata
    triple_vectors_sum['num_chiral_centers'] = len(chiral_atoms)
    
    return triple_vectors_sum

def process_sdf_file(sdf_filename, output_csv, max_bond_distance=None):
    """
    Process SDF file and calculate signed triple vectors for all molecules
    
    Args:
        sdf_filename: Input SDF file
        output_csv: Output CSV file
        max_bond_distance: Maximum bond distance from chiral centers (None for all atoms)
    """
    results = []
    
    # Read SDF file
    suppl = Chem.SDMolSupplier(sdf_filename, removeHs=False)
    
    for idx, mol in enumerate(suppl):
        if mol is None:
            print(f"Warning: Could not read molecule {idx}")
            continue
        
        try:
            # Get molecule ID
            if mol.HasProp("_Name"):
                mol_id = mol.GetProp("_Name")
            else:
                mol_id = f"mol_{idx}"
            
            print(f"Processing molecule {idx}: {mol_id}")
            
            # Calculate triple vectors with chiral-centered approach
            triple_vectors = calculate_all_triple_vectors(mol, max_bond_distance)
            
            # Prepare result row
            row = {'molecule_id': mol_id}
            row.update(triple_vectors)
            results.append(row)
            
        except Exception as e:
            print(f"Error processing molecule {idx}: {str(e)}")
            continue
    
    # Create DataFrame and save to CSV
    if results:
        df = pd.DataFrame(results)
        df.iloc[:, 1:-1].to_csv(output_csv, index=False)
        print(f"\nSuccessfully processed {len(results)} molecules")
        print(f"Results saved to: {output_csv}")
        
        # Print column information
        print(f"\nGenerated {len([k for k in triple_vectors.keys() if k.startswith('STP_')])} SIGNED triple vector descriptors:")
        print(f"Properties used: mass, gasteiger_charge, abs_gasteiger_charge, mmff_charge, abs_mmff_charge, polarizability, crippen_logP, crippen_MR, tpsa, labute_asa, estate")
        print(f"Total unique triple combinations: {len([k for k in triple_vectors.keys() if k.startswith('STP_')])}")
        if max_bond_distance is not None:
            print(f"Atoms restricted to within {max_bond_distance} bonds from chiral centers")
            
    else:
        print("No molecules were successfully processed")

def main():
    if len(sys.argv) < 3:
        print("Usage: python cSTP.py input.sdf output.csv [max_bond_distance]")
        print("")
        print("Arguments:")
        print("  input.sdf: Input SDF file with 3D molecular structures")
        print("  output.csv: Output CSV file with calculated descriptors")
        print("  max_bond_distance: (Optional) Maximum bond distance from chiral centers")
        print("                    If not specified, all atoms are used")
        print("")
        print("Example:")
        print("  python cSTP.py molecules.sdf descriptors.csv")
        print("  python cSTP.py molecules.sdf descriptors.csv 3")
        sys.exit(1)
    
    sdf_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Parse optional max_bond_distance parameter
    max_bond_distance = None
    if len(sys.argv) > 3:
        try:
            max_bond_distance = int(sys.argv[3])
            if max_bond_distance < 0:
                print("Error: max_bond_distance must be non-negative")
                sys.exit(1)
        except ValueError:
            print("Error: max_bond_distance must be an integer")
            sys.exit(1)
    
    print("CHIRAL ATOM-CENTERED SIGNED Scalar Triple Vector Calculator")
    print("======================================================")
    print("Features:")
    print("- Calculations centered on chiral atoms")
    print("- Sums triple vectors for all chiral centers")
    print("- Optionally restricts to atoms within specified bond distance from chiral centers")
    print("- Signed scalar triple products")
    print("- 11 atomic properties including:")
    print("  mass, gasteiger_charge, abs_gasteiger_charge, mmff_charge, abs_mmff_charge,")
    print("  polarizability, crippen_logP, crippen_MR, tpsa, labute_asa, estate")
    print(f"Input SDF: {sdf_file}")
    print(f"Output CSV: {output_file}")
    if max_bond_distance is not None:
        print(f"Max bond distance from chiral centers: {max_bond_distance}")
    else:
        print("Using all atoms (no distance restriction)")
    print()
    
    process_sdf_file(sdf_file, output_file, max_bond_distance)

if __name__ == "__main__":
    main()
