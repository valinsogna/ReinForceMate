#!/bin/bash

# Ask the user for the number of GPUs
echo "Please enter the number of GPUs or press enter to use none:"
read ngpus

# Ask the user for the number of CPUs per task
echo "Please enter the number of CPUs per task:"
read ncpus

# Ask the user for the memory needed
echo "Please enter the memory required (in GB):"
read mem

# Ask the user for the time needed
echo "Please enter the time required (in the format HH:MM:SS):"
read time_needed

# Ask the user for the partition
echo "Please enter the partition to use:"
read partition

# Check if the inputs are valid
if ! [[ "$ngpus" =~ ^[0-9]+$ ]] || ! [[ "$ncpus" =~ ^[0-9]+$ ]] || ! [[ "$mem" =~ ^[0-9]+$ ]] || ! [[ "$time_needed" =~ ^([0-9]+):([0-5][0-9]):([0-5][0-9])$ ]] || [[ -z "$partition" ]]
then
    echo "Error: Invalid input(s)"
    exit 1
fi

# Create a temporary job script
jobscript=$(mktemp)

# Write the job script
cat << EOF > $jobscript
#!/bin/bash
#SBATCH --job-name=ReinForceMate
#SBATCH --output=ReinForceMate.out
#SBATCH --partition=$partition
#SBATCH --nodes=1
#SBATCH --tasks=1
$gpu_string
#SBATCH --cpus-per-task=$ncpus
#SBATCH --mem=${mem}gb
#SBATCH --time=$time_needed

# Run the script with different parameters
srun python3 script.py --gamma=0.95 --lr=0.01 --iter=700
srun python3 script.py --gamma=0.5 --lr=0.01 --iter=700
srun python3 script.py --gamma=0.1 --lr=0.01 --iter=700

EOF

# Submit the job script
sbatch $jobscript

# Clean up the job script
rm $jobscript
