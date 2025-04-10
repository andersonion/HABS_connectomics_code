import os
from nipype.interfaces.dcm2nii import Dcm2niix
sbatch_folder_path = "/mnt/newStor/paros/paros_CT/ADNI3_0_4_years/unzipped/ADNI/sbatch/"
GD = "/home/apps/gunnies/"

def convert_dcm_to_nifti(input_dir, output_dir):
    # Check if the output directory exists, create if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    i =1
    for root, dirs, files in os.walk(input_dir):

        # Check if the current directory contains DICOM files
        if any(file.endswith('.dcm') for file in files):
            # Generate output folder structure to maintain the same as input directory
            relative_path = os.path.relpath(root, input_dir)
            output_subdir = os.path.join(output_dir, relative_path)

            # Ensure the output subdirectory exists
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            job_name = "A"+str(i)
            if (True):
                # Set up Dcm2niix conversion
                converter = Dcm2niix()
                converter.inputs.source_dir = root
                converter.inputs.output_dir = output_subdir
                converter.inputs.compress = 'y'  # Compress output files with gzip
                command1 = "hostname\n" + '/usr/bin/dcm2niix -b y -z y -x n -t n -m n -o ' + output_subdir+ ' -s n -v n ' +output_subdir
                command = GD + "submit_slurm_cluster_job.bash " + sbatch_folder_path + " "+ job_name + " 0 0 '"+ command1+"'"   

                
                
                os.system(command)
                #result = converter.run()
                #print(f"Converted DICOM files in {root} to NIfTI format in {output_subdir}")
            #except Exception as e:
                #print(f"Failed to convert DICOM files in {root}: {e}")
        i=i+1
# Define your input and output directories
#input_directory = '/mnt/newStor/paros/paros_CT/ADNI3_0_4_years/unzipped/ADNI/'
input_directory ='/mnt/newStor/paros/paros_CT/ADNI3_0_4_years/addon_110724/ADNI_2/'
output_directory = '/mnt/newStor/paros/paros_CT/ADNI3_0_4_years/addon_110724/ADNI_2/'

convert_dcm_to_nifti(input_directory, output_directory)
