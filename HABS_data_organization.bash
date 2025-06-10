#! /bin/env bash

# Basic preprocessing a data organization of HABS data, hopefully will be portable enough to be adapted for other studies.
parent_dir=$1;

# Assume the first arg is the directory containing the converted niftis
# Make sure there is a first arg:
if [[ -n ${parent_dir} ]];then
	# Test to see if first arg is an existing directory:
	if [[ -d ${parent_dir} ]];then
		# Set the study name, first looking at the second arg:
		if [[ -n ${2} ]];then
			study=$2
		else
			# If there is no second arg, assume the study is in the name of the $parent_dir:
			
			# First eat any and all trailing slashes:
			study=${parent_dir%%/};
			
			# Now eat anything up until and including the last slash:
			study=${study##*/};
			
			# Also assume that the study name is before any underscores:
			study=${study%%_*};			
		fi
		
	else
		# If not a directory as specified, assume it is a directory in the $WORK folder:
		test_dir=${WORK}/${parent_dir};
		
		if [[ -d ${test_dir} ]];then
			# If it does exist in the $WORK folder:
			
			# Assume the study name is before any underscores in the $parent_dir:
			# But it might be a sub-folder, so...
			# First eat any and all trailing slashes:
			study=${parent_dir%%/};
			
			# Now eat anything up until and including the last slash:
			study=${study##*/};
			
			# Also assume that the study name is before any underscores:
			study=${study%%_*};		
			
			# Reassign the parent directory to the proven-to-exist test directory
			parent_dir=${test_dir};
		else
			# Maybe it exists in the current directory?
			test_dir=${PWD}/${parent_dir};
			if [[ -d ${test_dir} ]];then
				# If it does exist in $PWD:
				
				# Assume the study name is before any underscores in the $parent_dir:
				# But it might be a sub-folder, so...
				# First eat any and all trailing slashes:
				study=${parent_dir%%/};
				
				# Now eat anything up until and including the last slash:
				study=${study##*/};
				
				# Also assume that the study name is before any underscores:
				study=${study%%_*};		
				
				# Reassign the parent directory to the proven-to-exist test directory
				parent_dir=${test_dir};
			else
				# Maybe $PWD should be the $parent_dir, but it's not worth figuring out a way
				# to safely test that assumption, since of course $PWD is guaranteed to exist
				# thus proving nothing. The user is on their own in this case, but at least be
				# nice enough to kindly let them know:
				echo "ERROR: No valid input directory found, and no further processing will happen."
				echo "If you want the current working directory to be the input directory,"
				echo "then please explicitly use it as your first argument."
				echo "(First argument you provided: $1)" && exit 1			
			fi

		fi

	fi
else
	# If no argument is fed to the script, then default to HABS data:
	parent_dir=${WORK}/HABS_niftis
	study=HABS
fi

# A little clean up in case any one is OCD
# Replace an instances of double slashes with a single slash:
parent_dir=${parent_dir//\/\//\/}

echo "Study name: ${study}"
echo "Inventorying and preprocessing data in: ${parent_dir}"

# Use a nickname:
pd=$parent_dir;

sbatch_dir=/${pd}/sbatch

if [[ ! -d $sbatch_dir ]];then
	mkdir $sbatch_dir;
fi

cd $pd;

# Assume only subject and sbatch folders in pd:
total_subs=$(ls -d */ 2>/dev/null | grep -v sbatch | wc -l)

if [[ ${total_subs} -gt 0 ]];then	
	echo "Total number of subjects: ${total_subs}"
else
	echo "ERROR: No subject folders seem to be found in ${pd},";
	echo "No further processing will take place..." && exit 1;
fi

# Count number of apparent subjects with DTI; change c_type and c_name as is appropriate for your data
c_type='*DTI*'
c_name='diffusion'
num_dwis=$(for runno in $(ls */${c_type}/ | grep ':' | cut -d '/' -f 1);do echo $runno;done | uniq| wc -l)
echo "Number of subjects with ${c_name} data: ${num_dwis}."

# Count number of apparent subjects with fMRI; change c_type and c_name as is appropriate for your data
c_type='*fMRI*'
c_name='fMRI'
num_fmris=$(for runno in $(ls */${c_type}/ | grep ':' | cut -d '/' -f 1);do echo $runno;done | uniq| wc -l)
echo "Number of subjects with ${c_name} data: ${num_fmris}."

# Count number of fMRI subjects with at least one diffusion:
c_type='*fMRI*'
c_name='fMRI'
d_type='*DTI*'
d_name='diffusion'
fmris=$(for runno in $(ls */${c_type}/ | grep ':' | cut -d '/' -f 1);do echo $runno;done | uniq);
bvals=$(ls */*/*/*/*bval)
d_nii_list=$pd/dwi_niis.txt

# Compiling list of 4D niis for diffusion data, and storing in a text file.
# This file should be deleted if you rerun with any data added or removed.
if [[ ! -f ${d_nii_list} ]];then
	echo "Compiling list of 4D diffusion niftis..."
	for bval in ${bvals};do
		test=$(fslhd ${bval/bval/nii.gz} | grep dim4 | head -1 | tr -s [:space:] ':' | cut -d ':' -f2);
		if [[ ${test} -gt 5 ]];then
			echo ${bval/bval/nii.gz} >> $d_nii_list;
		fi
	done
	echo "Done compiling list of 4D diffusion niftis."
else
	echo "NOTE: List of 4D diffusion niftis exists and won't be recompiled'."
	echo "File: ${d_nii_list}"
fi

only_fmri=$(for subject in $fmris;do test=$(ls */${d_type}/ 2>/dev/null | grep ':' | cut -d '/' -f 1 | wc -l);if ((! $test));then echo $subject;fi;done | wc -l)
echo "Number of subjects with only ${c_name} data (no ${d_type}): ${only_fmri}"

