#!/bin/bash
#SBATCH
#SBATCH --reservation=
#SBATCH  --mem=30000
#SBATCH  -v
#SBATCH  -s
#SBATCH  --output=/mnt/newStor/paros/paros_WORK/mouse/VBM_25HABS01_IITmean_RPI-results/papertrail/sbatch/slurm-%j.out
project=HABS
sbatch_dir="${BIGGUS_DISKUS}/VBM_25${project}01_IITmean_RPI-results/papertrail/sbatch/";

for runno in $(ls $BIGGUS_DISKUS/VBM_25${project}01_IITmean_RPI-work/preprocess/*fa_masked.nii.gz | cut -d '/' -f10 | cut -d 'f' -f1);do
	runno=${runno%_};
	job_name=${runno}_backport_atlas_labels;
	cmd="/mnt/newStor/paros/paros_WORK/${project}_connectomics/${project}_connectomics_code/create_backported_labels_for_${project}_fiber_tracking.bash ${runno}";
	bash $GUNNIES/submit_slurm_cluster_job.bash ${sbatch_dir} ${job_name} 0 0 $cmd;
done
   
