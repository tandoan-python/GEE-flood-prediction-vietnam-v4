import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import ee
import threading
import time
import os
import pandas as pd
import datetime
import traceback 

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
    # Đặc trưng địa hình
    'elevation', 'slope', 'aspect',
    # Đặc trưng lớp phủ và đất
    'land_cover', 'soil_type',
    # Flags từ land_cover
    'is_flood_prone', 'is_permanent_water', 'is_urban', 'is_agriculture',
    # Đặc trưng động
    'precip_total', 'precip_14_day', 'precip_7_day', 'precip_3_day',
    'soil_moisture'
]

# Simple in-memory TTL cache for GEE point queries to reduce latency
GEE_CACHE = {}
GEE_CACHE_LOCK = threading.Lock()
GEE_CACHE_TTL = 300  # seconds


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
    """Get GEE-derived features at a point with simple in-memory caching.

    Caches results for `GEE_CACHE_TTL` seconds keyed by rounded lat/lon to
    avoid frequent Earth Engine round-trips for nearby clicks.
    """
    # Coarse rounding to reuse nearby queries
    key = f"{round(lat,4)}_{round(lon,4)}"
    now_ts = time.time()

    with GEE_CACHE_LOCK:
        entry = GEE_CACHE.get(key)
        if entry:
            ts, data = entry
            if now_ts - ts < GEE_CACHE_TTL:
                # return a copy to avoid accidental mutation
                return dict(data)

    point = ee.Geometry.Point(lon, lat)
    today = ee.Date(datetime.datetime.now(datetime.timezone.utc))

    # --- 1. Static features ---
    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem).rename('slope')
    aspect = ee.Terrain.aspect(dem).rename('aspect')
    land_cover = ee.ImageCollection("ESA/WorldCover/v100").first().select('Map').rename('land_cover')
    soil = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b0').rename('soil_type')

    is_flood_prone = land_cover.eq(40).Or(land_cover.eq(50)).Or(land_cover.eq(90)).rename('is_flood_prone')
    is_permanent_water = land_cover.eq(80).rename('is_permanent_water')
    is_urban = land_cover.eq(50).rename('is_urban')
    is_agriculture = land_cover.eq(40).rename('is_agriculture')

    static_features_image = dem.rename('elevation').addBands([
        slope, aspect, land_cover.toByte(), soil.toByte(),
        is_flood_prone.toByte(), is_permanent_water.toByte(), is_urban.toByte(), is_agriculture.toByte()
    ])

    # --- 2. Antecedent / dynamic features ---
    end_date = today.advance(-3, 'day')

    pre_start_date_14 = end_date.advance(-14, 'day')
    precip_14_day = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").filterDate(pre_start_date_14, end_date).select('precipitation').sum().rename('precip_14_day')

    pre_start_date_7 = end_date.advance(-7, 'day')
    precip_7_day = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").filterDate(pre_start_date_7, end_date).select('precipitation').sum().rename('precip_7_day')

    pre_start_date_3 = end_date.advance(-3, 'day')
    precip_3_day = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").filterDate(pre_start_date_3, end_date).select('precipitation').sum().rename('precip_3_day')

    pre_start_date_3_sm = end_date.advance(-3, 'day')
    sm_collection = ee.ImageCollection("NASA/SMAP/SPL3SMP_E/005").filterDate(pre_start_date_3_sm, end_date).select('soil_moisture_am')

    collection_size = sm_collection.size()
    mean_sm_with_data = sm_collection.mean().unmask(0).rename('soil_moisture')
    mean_sm_empty = ee.Image(0).rename('soil_moisture')
    soil_moisture_mean = ee.Image(ee.Algorithms.If(collection_size.gt(0), mean_sm_with_data, mean_sm_empty))

    dynamic_features = precip_14_day.addBands([precip_7_day, precip_3_day, soil_moisture_mean, ee.Image(0).rename('precip_total')])

    # --- 3. Merge and fetch ---
    all_features_image = static_features_image.addBands(dynamic_features)
    data_dict = all_features_image.reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=90).getInfo()

    # Cache and return
    with GEE_CACHE_LOCK:
        GEE_CACHE[key] = (now_ts, dict(data_dict))

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
        
        # Trả về cả đặc trưng gốc để hiển thị
        features_dict = {}
        for col in df.columns:
            features_dict[col] = float(df[col].iloc[0])
        
        return {
            "probability": float(probability),
            "features": features_dict
        }

    except ee.ee_exception.EEException as e:
        raise HTTPException(status_code=500, detail=f"Loi GEE: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loi server: {e}")

# GFS retrieval code removed — forecasts are not used anymore. Kept removed
# version out of the repo history; functionality replaced by 'no_rain_forecast'

@app.post("/forecast")
def get_precipitation_forecast(point_data: PointData):
    """
    Endpoint trả về:
    1. Dự báo lượng mưa 7 ngày tới (3h một lần)
    2. Dự báo nguy cơ ngập cho 7 ngày tới (mỗi ngày 1 dự báo)
    """
    try:
        # New behaviour: do not use external rainfall forecasts (GFS).
        # Instead, return a 7-day flood probability forecast using current
        # features only (no assumed additional rainfall).
        current_features = get_gee_features_at_point(point_data.lat, point_data.lon)

        # Ensure dataframe has required columns in the correct order
        df_current = pd.DataFrame([current_features], columns=FEATURES_ORDER)
        if df_current.isnull().values.any():
            df_current = df_current.fillna(0)

        # Prepare 7-day forecasts: using the same features (no rainfall forecast)
        forecasts = []
        now = datetime.datetime.now(datetime.timezone.utc)
        base_precip = float(current_features.get('precip_total', 0) or 0)

        for day_offset in range(7):
            date_key = (now + datetime.timedelta(days=day_offset)).date().isoformat()

            # Copy features and set precip_total to base_precip (no change)
            features = current_features.copy()
            features['precip_total'] = base_precip

            df = pd.DataFrame([features], columns=FEATURES_ORDER)
            if df.isnull().values.any():
                df = df.fillna(0)

            scaled = scaler.transform(df)
            probability = float(model.predict_proba(scaled)[0][1])

            forecasts.append({
                'date': date_key,
                'precipitation_mm_24hr': None,
                'flood_probability': probability
            })

        return {
            'forecast': forecasts,
            'rain_forecast_used': False,
            'detail': {
                'method': 'no_rain_forecast',
                'note': 'Flood probability computed using current features only; no rainfall forecast used.',
                'current_features': current_features
            }
        }
    
    except ee.ee_exception.EEException as e:
        raise HTTPException(status_code=500, detail=f"Loi GEE: {e}")
    except HTTPException as e:
        raise e
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

