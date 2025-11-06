import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import ee
import os
import pandas as pd
import datetime
import traceback # Them traceback de in loi chi tiet

# =============================================================================
# KHỞI TẠO APP VÀ GEE
# =============================================================================
app = FastAPI()

try:
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    print("Khoi tao GEE thanh cong.")
except ee.ee_exception.EEException:
    print("Vui long xac thuc GEE (ee.Authenticate()) truoc khi chay API.")
    pass

# =============================================================================
# TẢI MODEL VÀ SCALER
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models'))
MODEL_PATH = os.path.join(MODEL_DIR, 'flood_model.xgb')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Tai model va scaler thanh cong.")
except FileNotFoundError:
    print(f"LOI: Khong tim thay model tai {MODEL_PATH} hoac scaler tai {SCALER_PATH}")
    model = None
    scaler = None

# Day la thu tu dac trung ma model da hoc (rat quan trong)
FEATURES_ORDER = [
    'aspect', 'elevation', 'land_cover', 'precip_14_day', 'precip_3_day', 
    'precip_7_day', 'precip_total', 'slope', 'soil_moisture', 'soil_type'
]

# =============================================================================
# ĐỊNH NGHĨA MODEL INPUT
# =============================================================================
class PointData(BaseModel):
    lat: float
    lon: float

# =============================================================================
# CÁC HÀM LOGIC GEE (Lay du lieu qua khu)
# =============================================================================
def get_gee_features_at_point(lat, lon):
    """
    Lay 10 dac trung (tinh va dong) tai 1 diem de du doan.
    Day la logic GEE de lay du lieu QUA KHU (historical/antecedent).
    """
    point = ee.Geometry.Point(lon, lat)
    today = ee.Date(datetime.datetime.utcnow())
    
    # --- 1. DAC TRUNG TINH (Giong het prepare_data.py) ---
    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem).rename('slope')
    aspect = ee.Terrain.aspect(dem).rename('aspect')
    land_cover = ee.ImageCollection("ESA/WorldCover/v100").first().select('Map').rename('land_cover')
    soil = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b0').rename('soil_type')

    static_features_image = dem.rename('elevation').addBands([
        slope, aspect, land_cover.toByte(), soil.toByte()
    ])
    
    # --- 2. DAC TRUNG DONG (QUA KHU - ANTECEDENT) ---
    end_date = today.advance(-3, 'day') # Lui 3 ngay de dam bao co du lieu GPM/SMAP
    start_date = end_date.advance(-1, 'day') 

    pre_start_date_14 = end_date.advance(-14, 'day')
    precip_14_day = ee.ImageCollection("NASA/GPM_L3/IMERG_V07") \
                      .filterDate(pre_start_date_14, end_date) \
                      .select('precipitation').sum().rename('precip_14_day')

    pre_start_date_7 = end_date.advance(-7, 'day')
    precip_7_day = ee.ImageCollection("NASA/GPM_L3/IMERG_V07") \
                     .filterDate(pre_start_date_7, end_date) \
                     .select('precipitation').sum().rename('precip_7_day')
    
    pre_start_date_3 = end_date.advance(-3, 'day')
    precip_3_day = ee.ImageCollection("NASA/GPM_L3/IMERG_V07") \
                     .filterDate(pre_start_date_3, end_date) \
                     .select('precipitation').sum().rename('precip_3_day')

    pre_start_date_3_sm = end_date.advance(-3, 'day')
    sm_collection = ee.ImageCollection("NASA/SMAP/SPL3SMP_E/005") \
                        .filterDate(pre_start_date_3_sm, end_date) \
                        .select('soil_moisture_am') 
    
    collection_size = sm_collection.size()
    mean_sm_with_data = sm_collection.mean().unmask(0).rename('soil_moisture')
    mean_sm_empty = ee.Image(0).rename('soil_moisture')
    soil_moisture_mean = ee.Image(
        ee.Algorithms.If(collection_size.gt(0), mean_sm_with_data, mean_sm_empty)
    )

    dynamic_features = precip_14_day.addBands([
        precip_7_day,
        precip_3_day,
        soil_moisture_mean,
        ee.Image(0).rename('precip_total')
    ])

    # --- 3. GOP TAT CA & LAY DU LIEU ---
    all_features_image = static_features_image.addBands(dynamic_features)
    
    data_dict = all_features_image.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=90
    ).getInfo()
    
    return data_dict

# =============================================================================
# ENDPOINT 1: DU DOAN XAC SUAT NGAP (CHO DONG HO)
# =============================================================================
@app.post("/predict")
def predict_flood(point_data: PointData):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model hoac Scaler chua duoc tai.")

    try:
        features_dict = get_gee_features_at_point(point_data.lat, point_data.lon)
        df = pd.DataFrame([features_dict], columns=FEATURES_ORDER)
        
        if df.isnull().values.any():
            with pd.option_context('future.no_silent_downcasting', True):
                df = df.fillna(0).infer_objects(copy=False)
            print("Canh bao: GEE tra ve gia tri Null, dang dien gia tri 0.")

        scaled_features = scaler.transform(df)
        probability = model.predict_proba(scaled_features)[0][1]
        
        return {"probability": float(probability)}

    except ee.ee_exception.EEException as e:
        raise HTTPException(status_code=500, detail=f"Loi GEE: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loi server: {e}")

# =============================================================================
# ENDPOINT 2: DU BAO MUA 14 NGAY (CHO BIEU DO)
# =============================================================================
def get_gfs_forecast_at_point(lat, lon):
    """
    Lay du lieu DU BAO (forecast) 14 ngay tu GFS.
    *** SUA LOI: Thay 'total_precipitation_surface' bang 'precipitation_rate' ***
    """
    point = ee.Geometry.Point(lon, lat)
    
    start_time_buffer = datetime.datetime.utcnow() - datetime.timedelta(days=1)
    
    # === PHAN SUA LOI 1 ===
    # Doi ten band
    gfs_collection = ee.ImageCollection("NOAA/GFS0P25") \
                        .filterDate(start_time_buffer, datetime.datetime.utcnow() + datetime.timedelta(days=16)) \
                        .select('precipitation_rate') # <--- SUA LOI TAI DAY
    
    sorted_collection = gfs_collection.sort('system:time_start', False)
    
    try:
        collection_size = sorted_collection.size().getInfo()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loi GEE khi kiem tra GFS size: {e}")

    if collection_size == 0:
        raise HTTPException(status_code=404, detail="Khong tim thay du lieu du bao GFS.")
    
    latest_forecast_image = sorted_collection.first()
    forecast_start_time = latest_forecast_image.date()
    
    full_forecast_collection = gfs_collection.filterMetadata('system:time_start', 'equals', forecast_start_time.millis())

    def get_value_at_point(image):
        # === PHAN SUA LOI 2 ===
        # Doi ten band
        precip = image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=27830
        ).get('precipitation_rate') # <--- SUA LOI TAI DAY
        
        return ee.Feature(None, {
            'time': image.date().format(),
            'precipitation_rate_kg_m2_s': precip # <--- SUA LOI TAI DAY (doi ten key)
        })

    forecast_data = full_forecast_collection.map(get_value_at_point).getInfo()
    
    results = []
    for f in forecast_data['features']:
        props = f['properties']
        
        # === PHAN SUA LOI 3 ===
        # Doi ten key de doc
        if props['precipitation_rate_kg_m2_s'] is not None:
            # GFS du bao theo (kg/m^2/s). De doi sang mm cho 3 gio:
            # (kg/m^2/s) * (3600 s/hr) * (3 hr) = kg/m^2 (tuong duong mm)
            precip_mm = float(props['precipitation_rate_kg_m2_s']) * 3 * 3600
            results.append({
                "time": props['time'],
                "precipitation_mm_3hr": precip_mm
            })
    return results

@app.post("/forecast")
def get_precipitation_forecast(point_data: PointData):
    try:
        forecast = get_gfs_forecast_at_point(point_data.lat, point_data.lon)
        return {"forecast": forecast}
    except ee.ee_exception.EEException as e:
        raise HTTPException(status_code=500, detail=f"Loi GEE: {e}")
    except Exception as e:
        print(f"Loi Python/FastAPI trong /forecast: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Loi server: {e}")

# =============================================================================
# ENDPOINT 0: TRANG GOC (Chao mung)
# =============================================================================
@app.get("/")
def read_root():
    return {"message": "Chao mung den voi API Du bao Ngap lut. API dang hoat dong!"}

# =============================================================================
# CHAY APP
# =============================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

