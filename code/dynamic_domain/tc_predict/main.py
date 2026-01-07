import datetime
import os, shutil
import importlib.util

from libs.hust.libtcg_SliceWindow import SliceWindow

from libs.uet.utils import gen_csv
from libs.uet.utils import model_predict
from libs.uet.utils import csv2nc

config_path = r'config.py'
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# có thể sửa thêm code để ghi thêm ra log
if __name__ == "__main__":
    
    working_dir = config.WORKING_DIR
    
    # Chuẩn bị các param cho sliding window
    # SliceWindow thực hiện trượt từng ô lưới (pixel) trên ảnh input
    # nếu ô lưới đó nằm trong phạm vi lat_min,lat_max,lon_min,lon_max
    # thì sẽ lấy ô đó làm trung tâm, cắt 1 ảnh con có kích thước [lat_dim, lon_dim] (VD: 17x17), lưu vào thư mục tạm
    input_path = config.INPUT_FOLDER
    output_folder = config.OUTPUT_FOLDER
    lat_min = config.LAT_MIN
    lat_max = config.LAT_MAX
    lon_min = config.LON_MIN
    lon_max = config.LON_MAX
    data_type = config.DATA_TYPE
    lat_dim = float(config.LAT_DIM)
    lon_dim = float(config.LON_DIM)
    # n_step = config.N_STEP
    lat_step = config.LAT_STEP
    lon_step = config.LON_STEP
    time_to_run = config.TIME_TO_RUN

    # tạo thư mục temp chứa kq trung gian: các ảnh con slide window, file csv cho model, file csv model predict
    prediction_time = datetime.datetime.now()
    temp_path = output_folder + '/' + os.path.join('data', 'temp', f'{prediction_time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(temp_path)

    # trong thư mục temp tạo thư muc slice_windows
    slice_windows_path = os.path.join(temp_path, "slice_windows")
    os.makedirs(slice_windows_path)
    
    proc_count = 8
    subproc_count = 8

    print('Start sliding window')
    SliceWindow(
        data_type, input_path, slice_windows_path,
        lat_min,lat_max,lon_min,lon_max,
        float(lat_dim),
        float(lon_dim),
        lat_step,
        lon_step,
        proc_count,
        subproc_count,
        time_to_run,
    )

    #######################
    # dựa trên các ảnh tạo được trong thư mục slice window ->
    # tạo file csv chứa đường dẫn đến các ảnh con, mục đích để sử dụng cho dataloader
    # file csv gồm các trường: datetime point   path    step
    # datetime: thời gian của các file input, lấy từ tham số [time_to_run]
    # point: (điểm), tọa độ của điểm trung tâm ảnh con, dạng [lat_lon]
    # path: đường dẫn đến ảnh trong thư mục slice window
    # step: trong trường hợp không aggregate, step = 0
    # trong trường hợp có aggregate, cùng 1 datetime input sẽ có nhiều step khác nhau: 0, -1, -2, ...
    # xem, sửa code trong /libs/uet/utils.py
    run_opt = config.RUN_OPT
    agg_steps = config.AGG_STEPS
    csv_folder = temp_path + '/' + 'csv'
    os.makedirs(csv_folder)

    csv_path = csv_folder + '/data.csv'
    
    gen_csv(
        slice_windows_path,
        data_type,
        run_opt,
        agg_steps,
        time_to_run,
        csv_path
    )
    
    ########################
    # Load data, model, predict và lưu vào file csv
    # xem, sửa code trong /libs/uet/utils.py, /libs/uet/Model/dataset.py, /libs/uet/Model/model.py
    # model_path = config.MODEL_PATH
    # lead_time = config.LEAD_TIME

    # # file csv output, gồm các trường: input_time   predict_time    location    score
    # # input_time: thời gian của các file input
    # # predict_time: thời gian dự báo: predict_time = input_time + lead_time
    # # location: lat, lon
    # # score: ước tính của model
    # prediction_results_path = csv_folder + '/prediction_results.csv'

    # model_predict(
    #     csv_path,
    #     data_type,
    #     run_opt,
    #     agg_steps,
    #     model_path,
    #     lead_time,
    #     prediction_results_path,
    # )

    # ###################################
    # # từ file prediction_results tạo ra map
    # # với mỗi predict time gen 1 map
    # # xem, sửa code trong /libs/uet/utils.py
    # output_folder = config.OUTPUT_FOLDER
    # gen_pdf = config.GEN_PDF

    # csv2nc(
    #     prediction_results_path, output_folder,
    #     lat_min,lat_max,lon_min,lon_max,
    #     data_type,
    #     gen_pdf
    # )

    # ####################################
    # # xóa output trung gian hay không (trong thư mục temp)
    # if config.CLEAN_UP:
    #     shutil.rmtree(temp_path)



