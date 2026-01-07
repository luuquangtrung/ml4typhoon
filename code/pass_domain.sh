python preprocessing\Preprocess_Ibtracs_Merra2.py
python past_domain\prepare_csv.py
python main.py --time t2_rus4_cw3_fe --norm_type new --lr 1e-7 --pos_ind 2 --under_sample --rus 4 --class_weight 3
python eval_fullyear.py --timestep t2_rus4_cw3_fe --model_path /past_domain/result/model/model-paht --fullmonth
python past_domain\seasonal_curve.py