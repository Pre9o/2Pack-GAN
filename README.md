# 2Pack-GAN: Exploring Transfer Learning to Fine-Tune Generative Adversarial Networks for Network Packet Generation

[![IEEE](https://img.shields.io/badge/IEEE-11073639-blue.svg)](https://ieeexplore.ieee.org/document/11073639)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)

A Wasserstein GAN (WGAN) implementation for generating synthetic network packets with support for transfer learning and fine-tuning. This repository contains the code for the paper "2Pack-GAN: Exploring Transfer Learning to Fine-Tune Generative Adversarial Networks for Network Packet Generation".

## Authors

- **Luiz A.C. Bianchi** - Federal University of Santa Maria (UFSM), Santa Maria, RS, Brazil
- **Rafael C. Pregardier** - Federal University of Santa Maria (UFSM), Santa Maria, RS, Brazil
- **Luis A. L. Silva** - Federal University of Santa Maria (UFSM), Santa Maria, RS, Brazil
- **Carlos R. P. dos Santos** - Federal University of Santa Maria (UFSM), Santa Maria, RS, Brazil

## Overview

2Pack-GAN uses Wasserstein GAN with Gradient Penalty (WGAN-GP) to generate realistic synthetic network packets. The model converts packet data into 28×28 matrix representations and learns to generate new packets that maintain the statistical properties of real network traffic.

### Key Features

- **WGAN-GP Architecture**: Stable training with Wasserstein loss and gradient penalty
- **Transfer Learning**: Fine-tune pre-trained models on new datasets
- **Layer Freezing**: Selective layer freezing during fine-tuning
- **Multiple Protocols**: Support for ICMP, DNS, and UDP packet generation
- **PCAP Export**: Direct conversion of generated packets to PCAP format
- **Real-time Monitoring**: TensorBoard integration for training visualization

## Project Structure

```
2Pack-GAN/
├── data/
│   ├── pcap_to_npz.py      # Convert PCAP files to NPZ datasets
│   ├── csv/                 # Domain lists for traffic generation
│   └── npz/                 # Processed datasets
├── pcaps/                   # Raw PCAP files
├── results/                 # Training outputs
│   └── <dataset_timestamp>/
│       ├── models/          # Saved generator models (.keras)
│       ├── weights/         # Model weights (.h5)
│       ├── output_images/   # Generated packet visualizations
│       ├── loss/            # Training history and loss plots
│       ├── logs/            # TensorBoard logs
│       └── synthetic_packets/ # Generated PCAP files
├── scripts/
│   └── dataset_generator.py # Capture real network traffic
└── src/
    ├── training.py          # Main training script
    ├── packets_generation.py # Generate synthetic packets
    ├── WGAN.py              # WGAN-GP implementation
    ├── models.py            # Generator and discriminator models
    ├── decoder.py           # Packet decoder
    └── data_loader.py       # Dataset loader
```

## Installation

### Requirements

- Python 3.8 or higher
- TensorFlow 2.13 or higher
- CUDA-compatible GPU (optional, but recommended)
- Root privileges (for packet capture only)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/2Pack-GAN.git
cd 2Pack-GAN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p data/npz pcaps results
```

## Usage

### 1. Dataset Preparation

#### Option A: Capture Real Traffic

Use the dataset generator script to capture ICMP and/or DNS traffic:

```bash
# Capture ICMP traffic
sudo python scripts/dataset_generator.py \
    --csv_name top500Domains.csv \
    --protocol icmp \
    --num_requests 5 \
    --pcap_name icmp_dataset \
    --interface eth0

# Capture DNS traffic
sudo python scripts/dataset_generator.py \
    --csv_name top500Domains.csv \
    --protocol dns \
    --num_requests 5 \
    --pcap_name dns_dataset \
    --interface eth0 \
    --dns_server 1.1.1.1

# Capture both ICMP and DNS
sudo python scripts/dataset_generator.py \
    --csv_name top500Domains.csv \
    --protocol icmp_dns \
    --num_requests 5 \
    --pcap_name combined_dataset \
    --interface eth0
```

#### Option B: Use Existing PCAP Files

Place your PCAP files in the `pcaps/` directory.

### 2. Convert PCAP to NPZ

Convert PCAP files to the NPZ format required for training:

```bash
python data/pcap_to_npz.py \
    --pcap_name your_dataset.pcap \
    --packets_limit 20000 \
    --protocol DNS \
    --npz_file your_dataset.npz
```

**Parameters:**
- `--pcap_name`: Name of the PCAP file
- `--packets_limit`: Maximum number of packets to process
- `--useful_data_start`: Start byte offset (default: 28, excludes Ethernet header)
- `--useful_data_end`: End byte offset (default: 300)
- `--protocol`: Protocol to filter (DNS, ICMP, UDP, or Other)
- `--npz_file`: Output NPZ filename

### 3. Train from Scratch

Train a new GAN model:

```bash
python src/training.py \
    --dataset_name your_dataset \
    --epochs 50 \
    --batch_size 1024 \
    --learning_rate 0.0001 \
    --random_dim 1024 \
    --discriminator_extra_steps 5 \
    --gp_weight 10.0 \
    --l2_reg 2.5e-5 \
    --examples 10
```

**Key Parameters:**
- `--dataset_name`: Name of NPZ file (without extension)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for Adam optimizer
- `--random_dim`: Latent vector dimension
- `--discriminator_extra_steps`: Discriminator training steps per generator step
- `--gp_weight`: Gradient penalty weight
- `--l2_reg`: L2 regularization factor
- `--examples`: Number of example images to generate per epoch

### 4. Fine-Tuning (Transfer Learning)

Fine-tune a pre-trained model on a new dataset:

```bash
python src/training.py \
    --dataset_name new_dataset \
    --fine_tune \
    --pretrained_epoch 49 \
    --freeze_layers 2 \
    --epochs 30 \
    --batch_size 1024 \
    --learning_rate 0.00005
```

**Transfer Learning Parameters:**
- `--fine_tune`: Enable fine-tuning mode
- `--pretrained_epoch`: Epoch number to load from pre-trained model
- `--freeze_layers`: Number of initial layers to freeze (0 = no freezing)

**Note:** When using `--fine_tune`, ensure the pre-trained model weights are in the expected directory structure under `results/`.

### 5. Generate Synthetic Packets

Generate synthetic packets using a trained model:

```bash
python src/packets_generation.py \
    --results_name your_dataset_20241210-120000 \
    --epoch 49 \
    --num_packets 1000
```

**Parameters:**
- `--results_name`: Name of the training results directory
- `--epoch`: Epoch number of the trained model
- `--num_packets`: Number of packets to generate

The output PCAP file will be saved to `results/<results_name>/synthetic_packets/`.

### 6. Monitor Training

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir results/<your_training_dir>/logs
```

Open your browser and navigate to `http://localhost:6006` to view:
- Generator and discriminator loss curves
- Loss difference (generator vs discriminator balance)
- Generated packet visualizations

## Model Architecture

### Generator

- **Input**: 1024-dimensional latent vector (random noise)
- **Architecture**:
  - Dense layers: 64 → 1024 → 12544 units
  - Reshape to 7×7×256
  - Conv2DTranspose: 64 filters (4×4 kernel, stride 2)
  - Conv2DTranspose: 32 filters (4×4 kernel, stride 2)
  - Conv2D: 1 filter (1×1 kernel, tanh activation)
- **Output**: 28×28×1 packet representation

### Discriminator

- **Input**: 28×28×1 packet representation
- **Architecture**:
  - Conv2D: 64 filters (4×4 kernel, stride 2)
  - Conv2D: 128 filters (4×4 kernel, stride 2)
  - Flatten
  - Dense: 1 unit (linear activation)
- **Output**: Wasserstein distance estimate

### Training Details

- **Loss Function**: Wasserstein loss with gradient penalty
- **Optimizer**: Adam (β₁ = 0.5, learning rate = 0.0001)
- **Regularization**: L2 weight regularization (2.5e-5)
- **Training Ratio**: 5 discriminator steps per generator step
- **Gradient Penalty Weight**: 10.0

## Packet Representation

Packets are encoded as 28×28 grayscale matrices:

1. **Extraction**: IPv4 and protocol headers extracted from PCAP
2. **Downscaling**: 2×2 blocks averaged to reduce dimensions
3. **Normalization**: Values normalized to [-1, 1] range for training
4. **Decoding**: Generated matrices decoded back to valid packet headers

The representation captures:
- IPv4 header (20 bytes)
- Protocol headers (UDP/ICMP/DNS)
- Protocol-specific data fields

## Output Files

Each training run creates a timestamped directory with:

```
results/<dataset_timestamp>/
├── hyperparameters.json       # Training configuration
├── models/
│   └── generator_model*.keras # Saved generator models
├── weights/
│   ├── generator/
│   │   └── generator_weights*.h5
│   └── discriminator/
│       └── discriminator_weights*.h5
├── output_images/
│   └── generated_images_*/    # Visualizations per epoch
│       ├── generated_image_*.png
│       └── generated_image_*.pcap
├── loss/
│   ├── training_history.json  # Raw training metrics
│   └── training_losses.png    # Loss curves
├── logs/                      # TensorBoard logs
└── synthetic_packets/         # Generated PCAP files
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@INPROCEEDINGS{11073639,
  author={Bianchi, Luiz A.C. and Pregardier, Rafael C. and Silva, Luis A. L. and dos Santos, Carlos R. P.},
  booktitle={NOMS 2025-2025 IEEE Network Operations and Management Symposium}, 
  title={2Pack-GAN: Exploring Transfer Learning to Fine-Tune Generative Adversarial Networks for Network Packet Generation}, 
  year={2025},
  volume={},
  number={},
  pages={1-9},
  keywords={Protocols;Transfer learning;Telecommunication traffic;Data collection;Generative adversarial networks;Data models;Tuning;Standards;Testing;Synthetic data;Generative Adversarial Network;Network Traffic Datasets;Synthetic Traffic Generation;Packet Generation;Controllable Generative Adversarial Network;Transfer Learning;Fine Tuning},
  doi={10.1109/NOMS57970.2025.11073639}}

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Troubleshooting

### GPU Issues

If TensorFlow doesn't detect your GPU:

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install CUDA-compatible TensorFlow
pip install tensorflow[and-cuda]
```

### Memory Errors

If you encounter out-of-memory errors:

- Reduce `--batch_size` (try 512 or 256)
- Enable GPU memory growth (already enabled in code)
- Reduce `--random_dim` (try 512)

### PCAP Capture Permissions

If packet capture fails:

```bash
# Grant capture capabilities to Python
sudo setcap cap_net_raw,cap_net_admin=eip $(which python3)

# Or run with sudo
sudo python scripts/dataset_generator.py ...
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors through the Federal University of Santa Maria (UFSM) or contact me at rcpregardier@inf.ufsm.br.

## Acknowledgments

This research was conducted at the Federal University of Santa Maria (UFSM), Santa Maria, RS, Brazil.