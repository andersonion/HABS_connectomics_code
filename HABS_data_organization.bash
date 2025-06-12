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

# Count number of fMRI subjects with at least one diffusion:
c_type='*fMRI*'
c_name='fMRI'
d_type='*DTI*'
d_name='diffusion'

#------

# Compiling list of 4D niis for fMRI data, and storing in a text file.
# This file should be deleted if you rerun with any data added or removed.
c_nii_list=$pd/${c_name}_niis.txt

# Bespoke to fMRI data, we test for a minumum of 7 volumes, but this can be adjusted
# for other types of multi-dimensional data...including DTI
more_vols_than=60;

all_c_niis=$(ls */${c_type}/*/*/*.nii.gz);


if [[ ! -f ${c_nii_list} ]];then
	echo "Compiling list of 4D ${c_name} niftis..."
	for nii in ${all_c_niis};do
		test=$(fslhd ${nii} | grep dim4 | head -1 | tr -s [:space:] ':' | cut -d ':' -f2);
		if [[ ${test} -gt ${more_vols_than} ]];then
			echo ${nii} >> $c_nii_list;
		fi
	done
	echo "Done compiling list of 4D ${c_name} niftis."
else
	echo "NOTE: List of 4D ${c_name} niftis exists and won't be recompiled'."
	echo "   File: ${c_nii_list}"
fi

c_subs=$(for nii in $(more ${c_nii_list});do echo ${nii%%/*};done | sort | uniq)
num_c=$(echo $c_subs | wc -w)
echo "Number of subjects with ${c_name} data: ${num_c}."

# Compiling list of 4D niis for diffusion data, and storing in a text file.
# This file should be deleted if you rerun with any data added or removed.
d_nii_list=$pd/${d_name}_niis.txt

# Bespoke to diffusion data, we test for a minumum of 7 volumes, but this can be adjusted
# for other types of multi-dimensional data...including fMRI
more_vols_than=6;

# For usable diffusion data, we need bvals/bvecs, so we look for either of those:
bvals=$(ls */*/*/*/*bval)


if [[ ! -f ${d_nii_list} ]];then
	echo "Compiling list of 4D ${d_name} niftis..."
	for bval in ${bvals};do
		test=$(fslhd ${bval/bval/nii.gz} | grep dim4 | head -1 | tr -s [:space:] ':' | cut -d ':' -f2);
		if [[ ${test} -gt ${more_vols_than} ]];then
			echo ${bval/bval/nii.gz} >> $d_nii_list;
		fi
	done
	echo "Done compiling list of 4D ${d_name} niftis."
else
	echo "NOTE: List of 4D ${d_name} niftis exists and won't be recompiled'."
	echo "   File: ${d_nii_list}"
fi

d_subs=$(for nii in $(more ${d_nii_list});do echo ${nii%%/*};done | sort | uniq)
num_d=$(echo $d_subs | wc -w)
echo "Number of subjects with ${d_name} data: ${num_d}."

#-----
# Test for anomolies with MORE than 2 dwis
anoms='';
for sub in ${d_subs};do
	test=$(grep "${sub}/${opt_proto_prefix}" $d_nii_list 2>/dev/null | wc -l) ;
	if [[ ${test} -gt 2 ]];then
		anoms="${anoms}${sub} ";
	fi
done

if [[ -n $anoms ]];then
	echo "Please inspect the following subjects, as they appear to have more than the expected maximum of 2 diffusion images:"
	echo "    ${anoms}"
fi

#-----

# How many subjects have both usable data for both fMRI and DtTI?

both_types=$(for subject in $c_subs;do echo $d_subs | grep ${subject} 2>/dev/null;done | wc -l)
echo "Number of subjects with both ${c_name} and ${d_name} data: ${both_types}"
#-----

# How many subjects only have raw fMRI data, and no usable DTI data?

# Note: We add a protocol prefix (which might change with study) to prevent catching randomly
# occurring strings elsewhere in file names
# Not needed now, but keeping the code for later...
opt_proto_prefix='/AX';

only_c=$(for subject in $c_subs;do echo $d_subs | grep ${subject} 2>/dev/null | wc -l);if ((! $test));then echo $subject;fi;done | wc -l)
echo "Number of subjects with only ${c_name} data (no ${d_name}): ${only_c}"
echo "Note that some subjects may have ${d_name} data, but are missing the raw 4D stack we want."

#-----

# How many subjects only have raw DTI data, and no usable fMRI data?

#opt_proto_prefix='/f';

only_d=$(for subject in $d_subs;do echo $c_subs | grep ${subject} 2>/dev/null | wc -l);if ((! $test));then echo $subject;fi;done | wc -l)
echo "Number of subjects with only ${d_name} data (no ${c_name}): ${only_d}"
echo "Note that some subjects may have ${c_name} data, but are missing the raw 4D stack we want."

#-----



#-----



#-----



#-----