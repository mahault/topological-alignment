# Remote Access: Spectrum (Alejandro's Machine)

## Machine Specs
- **GPU**: NVIDIA GeForce GTX 1650 (4GB VRAM)
- **CUDA**: 11.4, Driver 470.223.02
- **RAM**: 16GB
- **Disk**: 905GB (690GB free)
- **OS**: Ubuntu 18.04.6 LTS
- **Python**: 3.10.9 (via miniconda at `~/miniconda3/`)

## SSH Connection

### Via ngrok (current method)
Alejandro must start ngrok on his side first:
```bash
# On spectrum (Alejandro runs this):
ngrok tcp 22
```
This produces a forwarding address like `tcp://2.tcp.eu.ngrok.io:11665`.

Then connect from your machine:
```bash
ssh -p 11665 mao@2.tcp.eu.ngrok.io
```
- **User**: `mao`
- **Password**: `inference-priors-success`
- **SSH key**: Already installed (`~/.ssh/id_ed25519` on local -> `~/.ssh/authorized_keys` on spectrum)

**Note**: The ngrok address/port changes each time Alejandro restarts it. Ask him for the current address.

### Via Tailscale (requires admin on Windows)
If you have admin access or Tailscale installed:
```bash
ssh mao@100.75.146.30
# or
ssh mao@spectrum
```

## Project Layout on Spectrum
```
~/miniconda3/                  # Python 3.10 environment
~/belief-geodesics-paper/
  experiments/                 # All experiment code
    exp1_synthetic.py
    exp2_eeg.py
    exp3_social_media.py
    data_loaders.py
    plotting.py
  scripts/
    run_all.py
    precompute_embeddings.py
  figures/                     # Generated PDFs
```

## Running Experiments
```bash
# Use miniconda python:
cd ~/belief-geodesics-paper
~/miniconda3/bin/python scripts/run_all.py

# With real data:
~/miniconda3/bin/python scripts/run_all.py --real --data-dir data/

# Run in background (survives disconnect):
nohup ~/miniconda3/bin/python scripts/run_all.py > ~/experiment_run.log 2>&1 &

# Monitor:
tail -f ~/experiment_run.log
```

## Copying Files
```bash
# Local -> Spectrum:
scp -P 11665 local_file.py mao@2.tcp.eu.ngrok.io:~/belief-geodesics-paper/experiments/

# Spectrum -> Local:
scp -P 11665 mao@2.tcp.eu.ngrok.io:~/belief-geodesics-paper/figures/fig_exp1_combined.pdf ./

# Copy whole directory:
scp -P 11665 -r mao@2.tcp.eu.ngrok.io:~/belief-geodesics-paper/figures/ ./figures/
```

## Installed Packages
numpy, scipy, scikit-learn, matplotlib, networkx, giotto-tda,
sentence-transformers, torch (2.6.0 + CUDA 12.4), pandas, tqdm
