#!/bin/bash
#

export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
# specify the source and target directories
src_dir=/path/to/train/images
dest_dir=/lama/input
rgb_inter=lama/out_mask4
dest_dir2=lama/input2

for subdir in $src_dir/*; do
    if [ -d "$subdir" ]; then
        # extract the subdirectory name (e.g., 000001, 000002, etc.)
        subdir_name=$(basename $subdir)
	
	depth_dir=$subdir/mask_complete_10
	mkdir -p $dest_dir/$subdir_name
	echo "$depth_dir"
        for file in "$depth_dir"/*.png
	do
	  base_name=$(basename "$file")
	  if [[ $file == *_mask001.png ]]; then
	    #base_name=$(basename "$file")
	    #newname=$(echo "$base_name" | sed 's/_mask001//')
	    #echo "$newname"
 	    cp "$depth_dir/$base_name" "$dest_dir/$subdir_name/$basename"
 	  fi
	done
	
	#copy rgb
	rgb_dir=$subdir/rgb

	for file in "$rgb_dir"/*.jpg
	do
	  base_name=$(basename "$file")
          newname=$(echo "$base_name" | sed 's/.jpg/.png/')
	  echo "$newname"
	  #cp "$file" "$dest_dir/$file"
	  convert "$rgb_dir/$base_name" "$dest_dir/$subdir_name/$newname"
	done


	#do inpainting
    fi
done

python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/input outdir=$(pwd)/out_mask4
	
for subdir in $src_dir/*; do
    if [ -d "$subdir" ]; then

    subdir_name=$(basename $subdir)
	depth_dir=$subdir/mask_complete_10
	mkdir -p $dest_dir2/$subdir_name
	echo "$depth_dir"
    for file in "$depth_dir"/*.png
	do
	  base_name=$(basename "$file")
	  if [[ $file == *_mask001.png ]]; then
	    #base_name=$(basename "$file")
	    #newname=$(echo "$base_name" | sed 's/_mask001//')
	    #echo "$newname"
 	    cp "$depth_dir/$base_name" "$dest_dir2/$subdir_name/$basename"
 	  fi
	done

    echo "$subdir_name"
	rgb_dir=$rgb_inter/$subdir_name
	echo "$rgb_dir"
	for file in "$rgb_dir"/*.png
	#copy to new folder and repeat inpainting
	do
	  base_name=$(basename "$file")
	  echo "$base_name"
	  newname=$(echo "$base_name" | sed 's/_mask001//')
	  echo "$newname"
	  cp "$rgb_dir/$base_name" "$dest_dir2/$subdir_name/$newname"
	done	

	#python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/input2 outdir=$(pwd)/out_mask10
	
	#copy results to rgb_inpainted
	
    fi
done

python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/input2 outdir=$(pwd)/out_mask10

#for file in "$src_dir"/*.png
#do
#  if [[ $file == *_mask001.png ]]; then
#    base_name=$(basename "$file")
#    newname=$(echo "$base_name" | sed 's/_mask001//')
#    mv "$file" "$dest_dir/$newname"
#  fi
#done
