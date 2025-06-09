import time
import argparse
from pyscf import gto, scf, dft

def parse_arguments():
    parser = argparse.ArgumentParser(description="Test mixed-precision density fitting calculations.")
    parser.add_argument("xyz_file", type=str, help="Path to the XYZ file containing atomic coordinates.")
    parser.add_argument("basis", type=str, help="Basis set type.")
    parser.add_argument("--calc_type", type=str, default="DFT", choices=["HF", "DFT"], help="Calculation type: HF or DFT.")
    parser.add_argument("--xc", type=str, default="b3lyp", help="Exchange-correlation functional for DFT (default: b3lyp).")
    parser.add_argument("--mxp_df_level", type=int, default=4, choices=range(0, 5), help="MxP DF level (0, 1, 2, 3, 4, default: 4).")
    parser.add_argument("--mol_verbose", type=int, default=5, help="Verbosity level for molecule information (default: 5).")
    parser.add_argument("--run_verbose", type=int, default=5, help="Verbosity level for SCF run (default: 5).")
    parser.add_argument("--conv_tol", type=float, default=1e-7, help="SCF convergence tolerance (default: 1e-7).")
    parser.add_argument("--max_cycle", type=int, default=50, help="Maximum SCF cycles (default: 50).")
    parser.add_argument("--max_memory", type=int, default=32768, help="Maximum memory in MB (default: 32768).")
    return parser.parse_args()

def read_xyz_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()[2:]  # Skip first two lines (atom count and comment)
    return "\n".join(lines)

def main():
    args = parse_arguments()

    # Initialize molecule
    mol = gto.Mole(max_memory=args.max_memory)
    mol.atom = read_xyz_file(args.xyz_file)
    mol.basis = args.basis
    mol.verbose = args.mol_verbose
    mol.build()

    # Select calculation type
    if args.calc_type == "HF":
        mf = scf.RHF(mol).density_fit().to_gpu()
    elif args.calc_type == "DFT":
        mf = dft.RKS(mol, xc=args.xc).density_fit().to_gpu()
    else:
        raise ValueError(f"Unsupported calculation type: {args.calc_type}")

    # Set SCF options
    mf.conv_tol = args.conv_tol
    mf.max_cycle = args.max_cycle
    mf.verbose = args.run_verbose
    mf.mxp_df_level = args.mxp_df_level
    print(f"\n!! Mixed precision density fitting level = {args.mxp_df_level}")

    # Run calculation
    start_t = time.time()
    mf.kernel()
    end_t = time.time()
    print(f'SCF kernel time: {end_t - start_t:.3f} sec')

if __name__ == "__main__":
    main()