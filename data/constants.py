CL_max=400
# Maximum nucleotide context length (CL_max/2 on either side of the 
# position of interest)
# CL_max should be an even number

SL=5000
# Sequence length of SpliceAIs (SL+CL will be the input length and
# SL will be the output length)

splice_table='hg38V46_splice_table.txt' #change it to hg38V46_splice_table.txt for hg38
ref_genome='hg38.fa' #change it to hg38.fa for hg38
# Input details

data_dir='./'
sequence='canonical_sequence_hg38.txt' #change name for hg38
# Output details

version = "hg38"
