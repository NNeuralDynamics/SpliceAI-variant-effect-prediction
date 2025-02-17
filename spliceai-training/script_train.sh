source constants.py

#$ -q gpu
#$ -cwd
#$ -N train_spliceai
#$ -e Logs/
#$ -o Logs/
#$ -l gpus=2
#$ -l h_vmem=500g
export LD_LIBRARY_PATH=~/cuda/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
python3 -u train_model.py $1 $2 > Outputs/SpliceAI_$version_${1}_c${2}.txt
