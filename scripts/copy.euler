#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=4096
#SBATCH --time=120:00:00
#SBATCH --job-name="copy"
#SBATCH --open-mode=truncate
#SBATCH --mail-type=END,FAIL

# Source and destination directories
source_directory="/cluster/work/sachan/txu/Typological_Universal"
destination_directory="/cluster/project/sachan/txu/Typological_Universal_new"

# Check if the source directory exists
if [ ! -d "$source_directory" ]; then
    echo "Source directory does not exist."
    exit 1
fi

# Check if the destination directory exists; if not, create it
if [ ! -d "$destination_directory" ]; then
    mkdir -p "$destination_directory"
fi

# Use rsync with progress information
rsync -ah --info=progress2 "$source_directory/" "$destination_directory/" | {
  IFS="%"
  while read -r line; do
    # Extract the percentage from rsync's output
    percentage="${line##* }"
    # Use the percentage to update the progress bar
    echo -ne "Progress: [$percentage]\r"
  done
  # Print a newline to move to the next line
  echo ""
}

# Check the rsync exit status
if [ $? -eq 0 ]; then
    echo "Directory copy completed successfully."
else
    echo "Directory copy encountered an error."
fi