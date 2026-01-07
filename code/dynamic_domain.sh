python preprocessing\Preprocess_Ibtracs_Merra2.py
python Prepare_data.py --path $path --step 2 --ratio 30 --dst $dst
python Train.py --inp_dir $inp_dir --out_dir $out_dir --weight 6 --map_path $map_path
python Map_eval.py --temp $temp --out $out
python Spatial_map.py --temp $temp --out $out