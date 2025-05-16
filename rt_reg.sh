# Base directory containing sub-*/ folders
GLMSINGLE_DIR="/home/kahwang/argon/data/ThalHi/GLMsingle"

# Loop over each subject folder
for subdir in "${GLMSINGLE_DIR}"/sub-*; do
  [ -d "$subdir" ] || continue
  
  subj_dir=$(basename "$subdir")        
  id=${subj_dir#sub-}                   
  echo "=== Processing ${id} ==="

  # define input, regressor, and output paths
  input_vol="${subdir}/${id}_TrialBetas.nii.gz"
  regressor="${subdir}/RT.1D"
  output_vol="${subdir}/${id}_TrialBetas_resRT.nii.gz"

  # sanity checks
  if [ ! -f "$input_vol" ]; then
    echo "Warning: input volume not found: $input_vol" >&2
    continue
  fi
  if [ ! -f "$regressor" ]; then
    echo "Warning: regressor file not found: $regressor" >&2
    continue
  fi

  # run 3dTproject to regress out RT and save the residuals
  3dTproject \
    -input "$input_vol" \
    -ort "$regressor" \
    -prefix "$output_vol"

  echo "Output written to ${output_vol}"
done