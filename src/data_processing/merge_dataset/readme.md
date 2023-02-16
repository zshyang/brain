# 2023-02-13

## validate the meshes are correct

validate.py is created

there are totaly 614 hipp.

validate.py will create obj file from the folder and just save the

# Merge ADNI and OASIS Folder

## Step 1: Save in Other Formats

`0_save_mfile_into_npz_json` is a folder that contains the code to save the data into npz and json format.

1. Run `scripts/call_batch.sh` submit 3000 jobs to the cluster. Each job will save one mfile into npz and json format.

the next step is to make the mesh water tight.remesh the surface. sample jfeature from the old ones.
And we will need to save all the data into one h5py file. Such that we could translate the model very quick.
