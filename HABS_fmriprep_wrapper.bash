#! /bin/env bash
study=HABS
cd ${WORK}/${study}_inputs/
BIDS_dir=${WORK}/human/${study}/${study}_BIDS
list=$(ls *fMRI_nii4D.nii.gz | cut -d 'f' -f1);
list=$(echo $list | tr [:space:] 'X');
list=${list//_X/ };
for runno in $list;do
	r2=${runno//_/};
	T1=${BIDS_dir}/sub-${r2}/anat/sub-${r2}_T1w.nii.gz
	if [[ -f ${T1} ]];then
		python3 /home/apps/human_fMRI_pipeline/ADRC_fmri_prep_pipeline/fmri_prep.py  ${runno}
	fi
done
	