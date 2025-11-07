import ee
import time
import traceback
import sys

# =============================================================================
# KHỞI TẠO VÀ CẤU HÌNH GEE
# Phần mày phải đăng kí 1 tài khoản GEE và xác thực trước khi chạy.
# Bạn có thể làm theo hướng dẫn tại: https://developers.google.com/earth-engine/getstarted
# Sau khi xác thực, bạn có thể chạy script này.
# =============================================================================
try:
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
except ee.ee_exception.EEException:
    print("Vui long xac thuc GEE (ee.Authenticate()) truoc khi chay.")
    sys.exit("Dung chuong trinh.")
print("Khoi tao GEE thanh cong.")

# =============================================================================
# DANH SÁCH SỰ KIỆN LŨ LỤT (15 sự kiện lịch sử)
# =============================================================================
FLOOD_EVENTS = [
    {
        "id": "FL_TRN_2020_10", "start": "2020-10-10", "end": "2020-10-20", "apex": "2020-10-13",
        "detail": "Lu lich su mien Trung (Dot 1)", "purpose": "training"
    },
    {
        "id": "FL_TRN_2016_12", "start": "2016-12-12", "end": "2016-12-18", "apex": "2016-12-14",
        "detail": "Lu lon Quang Nam, Binh Dinh", "purpose": "training"
    },
    {
        "id": "FL_TRN_2022_10_A", "start": "2022-10-09", "end": "2022-10-15", "apex": "2022-10-11",
        "detail": "Lu tai Da Nang (Dot 1)", "purpose": "training"
    },
    {
        "id": "FL_TRN_2018_11", "start": "2018-11-18", "end": "2018-11-22", "apex": "2018-11-20",
        "detail": "Lu Quang Nam, Quang Ngai", "purpose": "training"
    },
    {
        "id": "FL_TRN_2018_08", "start": "2018-08-10", "end": "2018-08-20", "apex": "2018-08-15",
        "detail": "Lu DBSCL 2018", "purpose": "training"
    },
    {
        "id": "FL_TRN_2019_09", "start": "2019-09-10", "end": "2019-09-20", "apex": "2019-09-15",
        "detail": "Lu DBSCL 2019", "purpose": "training"
    },
    {
        "id": "FL_TRN_2015_07", "start": "2015-07-26", "end": "2015-08-03", "apex": "2015-07-30",
        "detail": "Lu lich su Quang Ninh", "purpose": "training"
    },
    {
        "id": "FL_TRN_2017_08", "start": "2017-08-01", "end": "2017-08-07", "apex": "2017-08-03",
        "detail": "Lu quet Son La, Yen Bai", "purpose": "training"
    },
    {
        "id": "FL_TRN_2018_06", "start": "2018-06-24", "end": "2018-06-28", "apex": "2018-06-26",
        "detail": "Lu quet Lai Chau, Ha Giang", "purpose": "training"
    },
    {
        "id": "FL_TRN_2021_08", "start": "2021-08-05", "end": "2021-08-10", "apex": "2021-08-07",
        "detail": "Lu DBSCL 2021", "purpose": "training"
    },
    # Tap VALIDATION (dung cho Optuna)
    {
        "id": "FL_VAL_2023_10", "start": "2023-10-25", "end": "2023-10-30", "apex": "2023-10-28",
        "detail": "Lu tai Da Nang, Hue (thang 10/2023)", "purpose": "validation"
    },
    {
        "id": "FL_VAL_2023_11", "start": "2023-11-13", "end": "2023-11-17", "apex": "2023-11-15",
        "detail": "Lu tai Quang Tri, Hue (thang 11/2023)", "purpose": "validation"
    },
    {
        "id": "FL_VAL_2020_07", "start": "2020-07-20", "end": "2020-07-25", "apex": "2020-07-22",
        "detail": "Lu Ha Giang, Cao Bang", "purpose": "validation"
    },
    # Tap TESTING (danh gia cuoi cung)
    {
        "id": "FL_TST_2022_10_B", "start": "2022-10-14", "end": "2022-10-18", "apex": "2022-10-16",
        "detail": "Lu lich su Da Nang (Dot 2)", "purpose": "testing"
    },
    {
        "id": "FL_TST_2024_06", "start": "2024-06-09", "end": "2024-06-11", "apex": "2024-06-10",
        "detail": "Lu quet Ha Giang (thang 6/2024)", "purpose": "testing"
    }
]

# =============================================================================
# ĐỊNH NGHĨA VÙNG QUAN TÂM (AOI)
# =============================================================================
AOI = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME', 'Viet Nam')).geometry()

# =============================================================================
# CÁC ĐẶC TRƯNG TĨNH (STATIC FEATURES) - DA DON GIAN HOA
# =============================================================================
"""
    Tạo một ảnh (ee.Image) chứa các đặc trưng tĩnh (không thay đổi theo thời gian) dùng cho mô hình
    
    Các tham số truyền vào hàm bao gồm:
    1. aoi (vùng quan tâm): Geometry của vùng cần phân tích (Việt Nam)
    2. dem (Digital Elevation Model): Đã giải thích trong hàm create_export_task
    3. slope (độ dốc): tính từ DEM, Đã giải thích trong hàm create_export_task
    
    Các đặc trưng tĩnh bao gồm:
    
    1. aspect (hướng sườn): tính từ DEM
    2. land_cover (bản đồ lớp phủ đất): sử dụng ESA WorldCover
    3. soil_type (loại đất): sử dụng OpenLandMap Soil Texture
    
    4. slope: biến truyền vào
    5. elevation (cao độ): biến truyền vào (DEM)
    
    Ảnh trả về được reproject về cùng hệ tọa độ của DEM với scale = 90 m.

"""
def get_static_features(aoi, dem, slope):
    """
    Dinh nghia cac dac trung tinh.
    LUU Y: DEM va Slope duoc truyen vao de tai su dung.
    """
    # 1. Aspect: Hướng sườn
    aspect = ee.Terrain.aspect(dem).rename('aspect')

    # 2. Land Cover
    # bản đồ lớp phủ đất thế giới (WorldCover v1.0), 
    # Band 'Map' mã hóa các lớp, tính bằng số nguyên thể hiện type: rừng, nước, built-up, cropland, v.v.).
    # Dùng để phân biệt vùng nước, đô thị, rừng, nông nghiệp… giúp mô hình phân biệt khả năng ngập.
    land_cover = ee.ImageCollection("ESA/WorldCover/v100").first() \
                   .select('Map') \
                   .clip(aoi).rename('land_cover')

    # 3. Soil Type
    soil = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02") \
             .select('b0') \
             .clip(aoi).rename('soil_type')

    # Tao cac flags huu ich tu land cover
    is_flood_prone = land_cover.eq(40).Or(land_cover.eq(50)).Or(land_cover.eq(90)).rename('is_flood_prone')
    is_permanent_water = land_cover.eq(80).rename('is_permanent_water')
    is_urban = land_cover.eq(50).rename('is_urban')
    is_agriculture = land_cover.eq(40).rename('is_agriculture')
    
    # Gop cac dac trung
    static_features_image = dem.rename('elevation').addBands([
        slope,
        aspect,
        land_cover.toByte(),
        soil.toByte(),
        # Them cac flags
        is_flood_prone.toByte(),
        is_permanent_water.toByte(),
        is_urban.toByte(),
        is_agriculture.toByte()
    ])

    # Chuan hoa scale (rat quan trong)
    static_features_image = static_features_image.reproject(
        crs=dem.projection().crs(),
        scale=30
    )
    
    return static_features_image

# =============================================================================
# CÁC ĐẶC TRƯNG ĐỘNG (DYNAMIC FEATURES)
# =============================================================================
def get_dynamic_features(start_date, end_date):
    """Dinh nghia cac dac trung dong (Giu nguyen logic da sua)"""
    gpm = ee.ImageCollection("NASA/GPM_L3/IMERG_V07") \
            .filterDate(start_date, end_date)
    total_precipitation = gpm.select('precipitation').sum().rename('precip_total')

    # 14-day antecedent
    pre_start_date_14 = ee.Date(start_date).advance(-14, 'day')
    pre_gpm_14 = ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
    precip_14_day = pre_gpm_14.filterDate(pre_start_date_14, start_date) \
                              .select('precipitation').sum().rename('precip_14_day')

    # 7-day antecedent
    pre_start_date_7 = ee.Date(start_date).advance(-7, 'day')
    pre_gpm_7 = ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
    precip_7_day = pre_gpm_7.filterDate(pre_start_date_7, start_date) \
                             .select('precipitation').sum().rename('precip_7_day')
    
    # 3-day antecedent
    pre_start_date_3 = ee.Date(start_date).advance(-3, 'day')
    pre_gpm_3 = ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
    precip_3_day = pre_gpm_3.filterDate(pre_start_date_3, start_date) \
                             .select('precipitation').sum().rename('precip_3_day')

    # Soil Moisture (Voi logic If/else chong loi)
    pre_start_date_3_sm = ee.Date(start_date).advance(-3, 'day')
    sm_collection = ee.ImageCollection("NASA/SMAP/SPL3SMP_E/005") \
                        .filterDate(pre_start_date_3_sm, start_date) \
                        .select('soil_moisture_am') 
    
    collection_size = sm_collection.size()
    mean_sm_with_data = sm_collection.mean().unmask(0).rename('soil_moisture')
    mean_sm_empty = ee.Image(0).rename('soil_moisture')
    
    soil_moisture_mean = ee.Image(
        ee.Algorithms.If(
            collection_size.gt(0),
            mean_sm_with_data,
            mean_sm_empty
        )
    )

    dynamic_features = total_precipitation.addBands([
        precip_14_day,
        precip_7_day,
        precip_3_day,
        soil_moisture_mean
    ])
    
    return dynamic_features

# =============================================================================
# === LOGIC TAO DU LIEU MOI (PHAT HIEN THAY DOI) ===
# =============================================================================
def get_flood_data(s1_baseline, s1_during, slope, features_to_add):
    """
    Tim kiem su thay doi (Change Detection) de xac dinh LU (flood=1)
    """
    diff = s1_baseline.subtract(s1_during).rename('s1_diff')
    
    # === THAY DOI QUAN TRONG ===
    # Nguong moi: diff.gt(2.5) -> Noi long de "nhay" hon voi lu nong/vua
    flood_map = diff.gt(2.5)
    
    # Chi lay mau o vung trung (slope < 20)
    flood_map = flood_map.And(slope.lt(20))
    
    flood_points = flood_map.selfMask().rename('flood') # Lay pixel co gia tri 1
    
    return flood_points.addBands(features_to_add)


def get_non_flood_data(s1_baseline, s1_during, slope, features_to_add):
    """
    Tim kiem vung KHONG THAY DOI (Non-change) de xac dinh KHONG LU (flood=0)
    (Giu nguyen logic)
    """
    diff = s1_baseline.subtract(s1_during).rename('s1_diff')
    
    # Vung khong lu la vung co su thay doi rat it (duoi 1.5dB)
    non_flood_map = diff.lt(1.5).And(slope.lt(20))
    
    # Tao mot anh toan so 0, ten la 'flood'
    non_flood_image = ee.Image(0).rename('flood')
    # Su dung non_flood_map lam mat na
    non_flood_points = non_flood_image.updateMask(non_flood_map)
    
    return non_flood_points.addBands(features_to_add)

# =============================================================================
# HÀM TẠO TASK XUẤT DỮ LIỆU (DA CAP NHAT LOGIC)
# =============================================================================
def create_export_task(aoi, event_dict, num_points=3000, scale=90):
    event_id = event_dict['id']
    start_date = ee.Date(event_dict['start'])
    end_date = ee.Date(event_dict['end'])
    
    print(f"Dang DINH NGHIA phep tinh cho su kien {event_id} (Phuong phap Change Detection)...")
    
    try:
        # === PHAN 1: TINH TOAN CAC DAC TRUNG CHUNG ===
        
        # 1.1. Tinh S1 Baseline (3 thang truoc lu)
        pre_end_date = start_date.advance(-1, 'day')
        pre_start_date = pre_end_date.advance(-3, 'month')
        
        s1_baseline = ee.ImageCollection('COPERNICUS/S1_GRD') \
                          .filterBounds(aoi) \
                          .filterDate(pre_start_date, pre_end_date) \
                          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
                          .select('VH').median().clip(aoi)

        # 1.2. Tinh S1 During (Trong khi lu)
        s1_during = ee.ImageCollection('COPERNICUS/S1_GRD') \
                        .filterBounds(aoi) \
                        .filterDate(start_date, end_date) \
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
                        .select('VH').mean().clip(aoi)
        
        # 1.3. Tinh DEM va Slope (chi 1 lan)
        # dem (Digital Elevation Model): Cao độ mặt đất so với mực nước biển, đơn vị là mét.
        dem = ee.Image("USGS/SRTMGL1_003").clip(aoi)
        
        # slope (độ dốc): Độ nghiêng của bề mặt đất, tính từ DEM.
        # Đơn vị là độ (0-90).
        # ee.Terrain.slope() sẽ trả về độ dốc theo đơn vị độ.
        slope = ee.Terrain.slope(dem).rename('slope')

        # 1.4. Goi cac ham dac trung
        static_features = get_static_features(aoi, dem, slope)
        dynamic_features = get_dynamic_features(event_dict['start'], event_dict['end'])
        
        all_features = static_features.addBands(dynamic_features)

        # === PHAN 2: TAO MAU (SAMPLING) ===
        
        # 2.1. Dinh nghia vung Lũ (flood=1)
        flood_data = get_flood_data(s1_baseline, s1_during, slope, all_features)
        
        # 2.2. Dinh nghia vung Không Lũ (flood=0)
        non_flood_data = get_non_flood_data(s1_baseline, s1_during, slope, all_features)

        # Kiem tra nhanh xem co du lieu khong (an toan)
        # .size() se bi loi neu collection rong, nen chung ta phai kiem tra s1_during
        s1_during_size = ee.ImageCollection('COPERNICUS/S1_GRD') \
                           .filterBounds(aoi) \
                           .filterDate(start_date, end_date) \
                           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
                           .size()

        if s1_during_size.getInfo() == 0:
            print(f"!!! Bo qua su kien {event_id} do thieu S1.")
            return None

        # Mau ngap (Positive = 1)
        flood_samples = flood_data.sample(
            region=aoi, 
            scale=scale,
            numPixels=num_points, 
            geometries=True, 
            tileScale=16 # Toi uu GEE
        )
        
        # Mau khong ngap (Negative = 0)
        non_flood_samples = non_flood_data.sample(
            region=aoi, 
            scale=scale,
            numPixels=num_points, 
            geometries=True, 
            tileScale=16
        )
            
        training_data = flood_samples.merge(non_flood_samples)
        
        # Them cac truong metadata
        training_data = training_data.map(lambda f: f.set({
            'event_id': event_id,
            'purpose': event_dict['purpose'],
            'apex_date': event_dict['apex'],
            'detail': event_dict['detail']
        }))
        
        # Dinh nghia Task
        task_description = f'export_flood_data_{event_id}'
        task = ee.batch.Export.table.toDrive(
            collection=training_data,
            description=task_description,
            folder='GEE_Flood_Exports', 
            fileNamePrefix=event_id,
            fileFormat='CSV'
        )
        return task
        
    except ee.ee_exception.EEException as e:
        # Bat loi .getInfo()
        print(f"!!! Loi GEE KHI DINH NGHIA su kien {event_id}: {e}")
        traceback.print_exc() 
        return None
    except Exception as e:
        print(f"!!! Loi Python KHI DINH NGHIA su kien {event_id}: {e}")
        traceback.print_exc() 
        return None

# =============================================================================
# HÀM CHẠY CHÍNH (Da cap nhat)
# =============================================================================

def main():
    print("Bat dau qua trinh CHUAN BI TASK XUAT DU LIEU (Phuong phap Change Detection)...")
    
    tasks_started = 0
    
    for event in FLOOD_EVENTS:
        task = create_export_task(
            aoi=AOI,
            event_dict=event,
            num_points=3000,
            scale=90          # Giu 90m de giam tai
        )
        
        if task:
            try:
                task.start()
                tasks_started += 1
                print(f"==> DA KHOI TAO TASK: {task.config['description']}")
                time.sleep(1) # Nghi 1 giay giua cac lan khoi tao task
            except ee.ee_exception.EEException as e:
                print(f"!!! KHONG THE KHOI TAO TASK cho {event['id']}: {e}")
        
        time.sleep(60) # Nghi 1 phút giua cac su kien

    print(f"\n==================================================================")
    print(f"HOAN TAT! Da khoi tao thanh cong {tasks_started} / {len(FLOOD_EVENTS)} tac vu (tasks).")
    print(f"BUOC TIEP THEO:")
    print(f"1. Mo GEE Code Editor, vao tab 'Tasks' va nhan 'Run' cho tat ca cac task moi.")
    print(f"2. Sau khi cac task hoan thanh, tai thu muc 'GEE_Flood_Exports' tu Google Drive.")
    print(f"3. Dat cac file CSV vao thu muc '../data/raw_exports/'.")
    print(f"4. Chay script 'python src/combine_data.py' de tong hop du lieu.")
    print(f"==================================================================")


if __name__ == "__main__":
    main()

