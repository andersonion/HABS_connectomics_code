#! /bin/env bash
study=HABS
BIDS_dir=${WORK}/human/${study}/${study}_BIDS/

if [[ ! -d ${BIDS_dir} ]];then
	mkdir -p ${BIDS_dir}
fi

cd ${WORK}/${study}_inputs/

list=$(ls *fMRI_nii4D.nii.gz | cut -d 'f' -f1);
list=$(echo $list | tr [:space:] 'X');
list=${list//_X/ };
for runno in $list;do
	python3 /home/apps/human_fMRI_pipeline/convert_to_BIDS.py ${WORK}/${study}_inputs/ ${BIDS_dir} ${runno}
done
	