import argparse
import subprocess
import sys
import os
import tempfile
from pathlib import Path

def create_slurm_script(just_command):
    """Create a SLURM batch script that runs the given just command."""
    
    job_name = f"just_{just_command.replace(' ', '_')}"
    
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=long
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:1
#SBATCH --constraint="vr40g|vr80g"
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Change to the working directory
cd {os.getcwd()}

# Run the just command
srun just {just_command}
discord {just_command} finished on $SLURM_JOB_ID
"""
    return slurm_script

def submit_job(just_command):
    """Submit a SLURM job for the given just command."""
    
    # Create SLURM script content
    script_content = create_slurm_script(just_command)
    
    # Write script to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    cmd = ['sbatch', script_path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    print(f"Job submitted: {result.stdout.strip()}")

def main():
    parser = argparse.ArgumentParser(description="Submit just commands as SLURM jobs")
    parser.add_argument('command', help='The just command to run')
    
    args = parser.parse_args()
    
    # Check if justfile exists
    if not Path('justfile').exists():
        print("Error: No justfile found in current directory", file=sys.stderr)
        sys.exit(1)
    
    submit_job(args.command)

if __name__ == '__main__':
    main()
