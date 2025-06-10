#! /bin/env bash

# Basic preprocessing a data organization of HABS data, hopefully will be portable enough to be adapted for other studies.
parent_dir=$1;

# Assume the first arg is the directory containing the converted niftis
# Make sure there is a first arg:
if [[ -n ${parent_dir} ]];then
	echo 1
	# Test to see if first arg is an existing directory:
	if [[ -d ${parent_dir} ]];then
	echo 2
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
	#parent_dir=${WORK}/HABS_niftis
	#study=HABS
	# Testing:
	parent_dir=youre/hilarious
	study=deez-eyeballs
fi

echo "Study name: ${study}"
echo "Inventorying and preprocessing data in: ${parent_dir}"