"""
Extract EEG dyad data from the Figshare .mat file into per-dyad numpy arrays.
Concatenates Recording01 (cooperation) + Recording02 (competition) for each dyad,
so condition varies WITHIN each dyad's time series.

Output structure:
    data/eeg_collab/dyad_XX/
        participant1.npy   (4, T) - channels x time
        participant2.npy   (4, T)
        conditions.npy     (T,)  - 0=cooperation (Recording01), 1=competition (Recording02)
"""

import numpy as np
from pathlib import Path

MAT_FILE = Path(__file__).parent.parent / "data" / "eeg_collab" / "dyad_00" / "EEG.mat"
OUT_DIR = Path(__file__).parent.parent / "data" / "eeg_collab"


def extract_dyads():
    import scipy.io

    print(f"Loading {MAT_FILE}...")
    mat = scipy.io.loadmat(str(MAT_FILE))
    eeg = mat['EEG']

    rec1 = eeg['Recording01'][0, 0]  # cooperation
    rec2 = eeg['Recording02'][0, 0]  # competition

    dyad_names = rec1.dtype.names  # Dyad01..Dyad08
    print(f"{len(dyad_names)} dyads found")

    for dyad_name in dyad_names:
        dyad_idx = int(dyad_name.replace('Dyad', '')) - 1

        # Extract cooperation data
        coop_p1_list, coop_p2_list = [], []
        dyad_coop = rec1[dyad_name][0, 0]
        for task_name in dyad_coop['P01'][0, 0].dtype.names:
            task = dyad_coop['P01'][0, 0][task_name][0, 0]
            coop_p1_list.append(task['data'][0, 0])
        for task_name in dyad_coop['P02'][0, 0].dtype.names:
            task = dyad_coop['P02'][0, 0][task_name][0, 0]
            coop_p2_list.append(task['data'][0, 0])

        # Extract competition data
        comp_p1_list, comp_p2_list = [], []
        dyad_comp = rec2[dyad_name][0, 0]
        for task_name in dyad_comp['P01'][0, 0].dtype.names:
            task = dyad_comp['P01'][0, 0][task_name][0, 0]
            comp_p1_list.append(task['data'][0, 0])
        for task_name in dyad_comp['P02'][0, 0].dtype.names:
            task = dyad_comp['P02'][0, 0][task_name][0, 0]
            comp_p2_list.append(task['data'][0, 0])

        # Concatenate cooperation tasks, then competition tasks
        p1_coop = np.concatenate(coop_p1_list, axis=1)
        p2_coop = np.concatenate(coop_p2_list, axis=1)
        p1_comp = np.concatenate(comp_p1_list, axis=1)
        p2_comp = np.concatenate(comp_p2_list, axis=1)

        # Match lengths (P1 and P2 might differ slightly)
        min_coop = min(p1_coop.shape[1], p2_coop.shape[1])
        min_comp = min(p1_comp.shape[1], p2_comp.shape[1])

        p1_coop = p1_coop[:, :min_coop]
        p2_coop = p2_coop[:, :min_coop]
        p1_comp = p1_comp[:, :min_comp]
        p2_comp = p2_comp[:, :min_comp]

        # Concatenate coop + comp
        p1_full = np.concatenate([p1_coop, p1_comp], axis=1)
        p2_full = np.concatenate([p2_coop, p2_comp], axis=1)
        conditions = np.concatenate([
            np.zeros(min_coop, dtype=float),
            np.ones(min_comp, dtype=float),
        ])

        # Save
        dyad_dir = OUT_DIR / f"dyad_{dyad_idx:02d}"
        dyad_dir.mkdir(exist_ok=True)

        np.save(str(dyad_dir / 'participant1.npy'), p1_full)
        np.save(str(dyad_dir / 'participant2.npy'), p2_full)
        np.save(str(dyad_dir / 'conditions.npy'), conditions)

        print(f"  {dyad_name}: coop={min_coop} samples ({min_coop/250:.0f}s), "
              f"comp={min_comp} ({min_comp/250:.0f}s), "
              f"total={p1_full.shape[1]} -> dyad_{dyad_idx:02d}/")


if __name__ == "__main__":
    extract_dyads()
    print("\nDone!")
