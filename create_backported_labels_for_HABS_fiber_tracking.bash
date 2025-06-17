#!/bin/bash
#SBATCH 
#SBATCH --reservation=
#SBATCH  --mem=30000 
#SBATCH  -v
#SBATCH  -s 
#SBATCH  --output=/mnt/newStor/paros/paros_WORK/mouse//VBM_25HABS01_IITmean_RPI-results/papertrail/sbatch/slurm-%j.out 
#$ -l h_vmem=30000M,vf=30000M 
#$ -N create_backported_labels
#$ -M rja20@duke.edu 
#$ -m ea 
#$ -o /mnt/munin6/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-results/papertrail/sbatch//slurm-$JOB_ID.out 
#$ -e /mnt/munin6/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-results/papertrail/sbatch/slurm-$JOB_ID.out
runno=$1;
#project=$2;
project=HABS
template_prefix="faMDT_NoNameYet_n491_i6"

echo "Backporting labels to raw space for runno: ${runno}";
out_dir="${BIGGUS_DISKUS}/VBM_25${project}01_IITmean_RPI-results/connectomics/${runno}/";
work="${BIGGUS_DISKUS}/VBM_25${project}01_IITmean_RPI-work/"
if [[ ! -d ${out_dir} ]];then
    mkdir -pm 775 ${out_dir};
fi
dirty_dir=$BIGGUS_DISKUS/burn_after_reading/
if [[ ! -d ${dirty_dir} ]];then
    mkdir -m 775 ${dirty_dir};
fi

sbatch_dir="/mnt/newStor/paros/paros_WORK/mouse//VBM_25${project}01_IITmean_RPI-results/papertrail/sbatch/"
if [[ ! -d ${sbatch_dir} ]];then
    mkdir -m 775 ${sbatch_dir};
fi

atlas=IITmean_RPI;
atlas_labels="${ATLAS_FOLDER}/${atlas}/${atlas}_labels.nii.gz";
sbatch_file=$(ls -t ${work}preprocess/sbatch/*_${runno}_fa_recentering_and_setting_image_orientation_to_*.bash | head -1);
o_orientation=$(grep -v '#' $sbatch_file | head -1 | tail -1 | cut -d ' ' -f4);
w_orientation=$(grep -v '#' $sbatch_file | head -1 | tail -1 | cut -d ' ' -f5);

trans="${work}/preprocess/base_images/translation_xforms/${runno}_0DerivedInitialMovingTranslation.mat"; 
rigid="${work}/dwi/${runno}_rigid.mat";
affine="${work}/dwi/${runno}_affine.mat";
MDT_to_runno="${work}/dwi/SyN_0p5_3_0p5_fa/${template_prefix}/reg_diffeo/MDT_to_${runno}_warp.nii.gz";
MDT_to_atlas_affine="${work}/dwi/SyN_0p5_3_0p5_fa/${template_prefix}/stats_by_region/labels/transforms/MDT_fa_to_${atlas}_affine.mat";
atlas_to_MDT="${work}/dwi/SyN_0p5_3_0p5_fa/${template_prefix}/stats_by_region/labels/transforms/${atlas}_to_MDT_warp.nii.gz";
w_ref="${work}/preprocess/${runno}_fa_masked.nii.gz";
w_labels="${dirty_dir}${runno}_${w_orientation}_labels.nii.gz";
fixed_w_labels="${dirty_dir}${runno}_fixed_${w_orientation}_labels.nii.gz";
o_labels="${dirty_dir}${runno}_${o_orientation}_labels.nii.gz";
spacing=$(PrintHeader ${work}/preprocess/${runno}_fa_masked.nii.gz 1);


# Jacques, here is where you need to tweak the code so it finds the right co-reg'd nii4D's.
#final_ref="/mnt/munin6/Badea/Lab/mouse/co_reg_LPCA_${runno:1:5}_m00-results/Reg_LPCA_${runno:1:5}_nii4D.nii.gz";
final_ref="$BIGGUS_DISKUS/../human/diffusion_prep_MRtrix_${runno}/${runno}_fa.nii.gz";

#symbolic_ref="${out_dir}/${runno}_Reg_LPCA_nii4D.nii.gz";
final_labels="${out_dir}/${runno}_${atlas}_labels.nii.gz";

echo "hiya final label $final_labels"

if [[ ! -f ${final_labels} ]];then
    if [[ ! -f ${w_labels} ]];then
		cmd="antsApplyTransforms -v 1 -d 3 -i ${atlas_labels} -o ${w_labels} -r ${w_ref} -n MultiLabel[${spacing},2] -t [${trans},1] [${rigid},1] [${affine},1] ${MDT_to_runno} [${MDT_to_atlas_affine},1] ${atlas_to_MDT}";
		#echo $cmd;
		$cmd;
    fi
	if [[ $w_orientation == $o_orientation ]]; then
		echo "Orientations are the same; doing less work than otherwise..."
		if [[ -f ${w_labels} && ! -f ${final_labels} ]];then
			cmd_2="${GUNNIES}/nifti_header_splicer.bash ${final_ref}  ${w_labels} ${final_labels}";
			echo $cmd_2;
			$cmd_2;
		fi
	else
	echo "Orientations are NOT the same; doing MORE work than otherwise..."
		if [[ -f ${w_labels} && ! -f ${fixed_w_labels} ]];then
			${GUNNIES}/nifti_header_splicer.bash ${final_ref}  ${w_labels} ${fixed_w_labels};
		fi
		
		if [[ -f ${fixed_w_labels} ]] && [[ ! -f ${o_labels} ]];then
			echo '${MATLAB_EXEC_PATH}/img_transform_executable/run_img_transform_exec.sh ${MATLAB_2015b_PATH} ${fixed_w_labels} ${w_orientation} ${o_orientation} ${o_labels};'
			${MATLAB_EXEC_PATH}/img_transform_executable/run_img_transform_exec.sh ${MATLAB_2015b_PATH} ${fixed_w_labels} ${w_orientation} ${o_orientation} ${o_labels};
		fi

		if [[ -f ${o_labels} ]];then
			${GUNNIES}/nifti_header_splicer.bash ${final_ref}  ${o_labels} ${final_labels};
		fi
	fi

    if [[ -f ${final_labels} ]];then
		fslmaths ${final_labels} -add 0 ${final_labels} -odt short;
    fi
fi


copy_labels_to_origin_dir=1;
if (($copy_labels_to_origin_dir)); then 
	copy=${final_ref%/*}/
	if [[ ! -f ${copy} ]];then
		cp ${final_labels} ${copy};
	fi
fi

skip_making_data_package_for_tractography=1;

if (( ! $skip_making_data_package_for_tractography ));then
    echo 'if you see this, make the bval fixes'
    if [[ ! -e ${symbolic_ref} ]];then
		ln -s ${final_ref} ${symbolic_ref};
    fi
    
    bvals="${BIGGUS_DISKUS}/diffusion_prep_${runno:1:5}/${runno:1:5}_bvals.txt";
    bvecs="${BIGGUS_DISKUS}/diffusion_prep_${runno:1:5}/${runno:1:5}_bvecs.txt";

    bval_copy="${out_dir}/${runno}_bvals.txt";
    bvec_copy="${out_dir}/${runno}_bvecs.txt";
    
    if [[ ! -f ${bval_copy} ]];then
	cp ${bvals} ${bval_copy};
    fi

    if [[ ! -f ${bvec_copy} ]];then
	cp ${bvecs} ${bvec_copy};
    fi
fi

