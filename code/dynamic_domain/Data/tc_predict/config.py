
from datetime import datetime, timedelta

# COMMON
WORKING_DIR = 'path-to-executing-dir' # Thư mục chạy code
INPUT_FOLDER = 'input-dir' # thư mục chứa ảnh input
OUTPUT_FOLDER = 'output-dir' # thư mục chứa ảnh output


# 0: no_aggregate; 1: aggregate;...
# RUN_OPT = 0 thì AGG_STEPS không sử dụng
RUN_OPT = 0
AGG_STEPS = 1 # số step aggregate, ví dụ steps = 5 sẽ aggregate [t0, t-1, t-2, t-3, t-4] -> predict cho 1 thời gian t + k nào đó trong tương lai

#DATA_TYPE 0:FNL; 1:MERRA2, tham số liên quan đến cách xử lý dữ liệu input
DATA_TYPE = 1

# Các tham số liên quan đến sliding windows
# [sliding_window]
# grid extent: tọa độ 4 góc của lưới output mong muốn
LAT_MIN = 0
LAT_MAX = 30
LON_MIN = 100
LON_MAX = 150

# LAT_DIM, LON_DIM: kích thước 2 chiều 1 ảnh con sẽ cắt từ ảnh input khi trượt cửa sổ. VD: 17x17
LAT_DIM = 33
LON_DIM = 33
# N_STEP: số bước nhảy khi trượt cửa sổ
N_STEP = 5 
LAT_STEP = 10
LON_STEP = 8

# TIME_TO_RUN: thời gian của các file input, có thể sửa code để đọc từ 1 file csv, thay vì khai báo mảng như này
TIME_TO_RUN = [
    '2002-09-12 06:00:00',
    '2003-09-12 00:00:00',
    '2004-09-11 18:00:00',
    '2005-09-11 12:00:00',
    '2006-09-11 06:00:00',
    '2007-09-11 00:00:00',
    '2008-09-10 18:00:00',
]

dt_iter = datetime(1980, 1, 1)
dt_stop = datetime(1986, 1, 1)
dt_step = timedelta(hours=3)

TIME_TO_RUN = []
while dt_iter < dt_stop:
    TIME_TO_RUN.append(dt_iter.strftime('%Y-%m-%d %H:%M:%S'))
    dt_iter += dt_step

# Các tham số liên quan đến mô hình, output
#[model]
# MODEL_PATH: đường dẫn đến mô hình đã lưu
MODEL_PATH = WORKING_DIR + '/data/model/modelFNL.pth'
# LEAD_TIME: thời gian dự báo của mô hình. VD: mô hình dự báo 6h sau đó. Tham số này được dùng để tạo ra tên file output
LEAD_TIME = 0 # hours

# GEN_PDF: có tạo file pdf map output hay không
GEN_PDF = True
# CLEAN_UP: có xóa thư mục temp chứa các output trung gian hay không
CLEAN_UP = False