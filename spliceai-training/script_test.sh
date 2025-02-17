source constants.py

#$ -q gpu
#$ -cwd
#$ -N test_spliceai
#$ -e Logs/
#$ -o Logs/
#$ -l gpus=1
#$ -l h_vmem=500g
# export LD_LIBRARY_PATH=~/cuda/lib64:$LD_LIBRARY_PATH
# source ~/.bashrc
python -u test_model.py $1 > Outputs/SpliceAI_test_$version_${1}.txt
