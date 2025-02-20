from Bio import SeqIO
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from utils import one_hot_encode_sequence  # Ensure utils is correctly imported
from constants import *  # Ensure constants like CL_max and SL are defined

# Define input/output directories
INPUT_DIR = "../data/sequences/"
OUTPUT_DIR = "sequence_output_predictions/"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists

# List and filter FASTA files in the directory
files = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]
print(f"Found {len(files)} FASTA files:", files)

# Load pre-trained SpliceAI models
print("Loading models...")
MODEL_VERSIONS = [1, 2, 3, 4, 5]
models = [load_model(f'../Models/pre-trained/spliceai{v}.h5') for v in MODEL_VERSIONS]

# Check GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
print(f"Number of physical GPUs: {len(physical_devices)}")

# Process each FASTA file
for fasta_file in files:
    fasta_path = os.path.join(INPUT_DIR, fasta_file)

    # Read sequence from FASTA file
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequence = str(record.seq)

    original_seq_len = len(sequence)
    output_central_len = original_seq_len - CL_max
    print(f"Processing: {fasta_file} | Sequence Length: {original_seq_len}")

    # Padding to make sequence length a multiple of SL
    num_points = len(sequence) // SL
    sequence += 'N' * ((num_points + 1) * SL - len(sequence))
    num_points = len(sequence) // SL

    # Split sequences into overlapping windows
    sequences_split = [sequence[SL * i: SL * (i + 1) + CL_max] for i in range(num_points - 2)]

    print(f"Split into {len(sequences_split)} sequences of length {len(sequences_split[0])}")

    # Run SpliceAI predictions
    outputs, central_sequences = [], []
    for seq in sequences_split:
        one_hot_seq = one_hot_encode_sequence(seq)[None, :]
        output = np.mean([model.predict(one_hot_seq) for model in models], axis=0)
        outputs.append(output)
        central_sequences.append(seq[CL_max // 2: CL_max])  # Extract central portion

    # Form long sequence and corresponding outputs
    long_sequence = ''.join(central_sequences)
    long_outputs = np.concatenate(outputs, axis=1)  # Shape: (1, total_seq_length, num_classes)

    # Ensure length consistency
    final_seq = long_sequence[:output_central_len]
    final_outputs = long_outputs[:, :output_central_len, :][0]

    assert len(final_seq) == final_outputs.shape[0], "Mismatch between sequence and output length!"

    # Create DataFrame
    df = pd.DataFrame({
        "nucleotide": list(final_seq),
        "neither_probability": final_outputs[:, 0],
        "acceptor_probability": final_outputs[:, 1],
        "donor_probability": final_outputs[:, 2]
    })

    # Save to CSV
    filename = "_".join(fasta_file.split("/")[-1].split(".")[:2]) + "_spliceai_predictions.csv"
    output_path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(output_path, index=False)

    print(f"CSV saved: {output_path} | Shape: {df.shape}")
