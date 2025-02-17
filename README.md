# variant effect prediction using SpliceAI
SpliceAI wrapper for retraining SpliceAI and variant effect prediction

Create environment as - 
```sh
python -m venv spliceai
source spliceai/bin/activate
pip install -r requirements.txt
```

We will require bedtools for grabing sequences to create the dataset that we can refer [here](https://bedtools.readthedocs.io/en/latest/content/installation.html) for installing it.

Firstly, Update the constants.py file:
- ref_genome: path of the genome.fa file (hg19/GRCh37) or (hg38.fa)
- splice_table: path for reference splicing sequences (canonical_dataset.txt for hg19 and hg38V46_splice_table.txt for hg38)
- sequence: for sequence name
- version: is used for naming the file

Then, download the appropriate genome FASTA file for your dataset.
```sh
cd data
```
For GRCh37/hg19
```sh
!wget http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
```
Or for GRCh38/hg38
```sh
!wget http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
```

Finally, use the following commands for data preprocessing:

```sh
cd data/
./grab_sequence.sh

python create_datafile.py train all
python create_datafile.py test 0

python create_dataset.py train all
python create_dataset.py test 0
```

use the following commands for training the models:

make sure constants.py has correct version set as hg19 or hg38
```sh
cd spliceai-training/
qsub script_train.sh 10000 1
qsub script_train.sh 10000 2
qsub script_train.sh 10000 3
qsub script_train.sh 10000 4
qsub script_train.sh 10000 5
```

use the following commands for training the models:

```sh
qsub script_test.sh 10000
```