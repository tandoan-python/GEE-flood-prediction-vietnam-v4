import pandas as pd
import glob
import os
import joblib
from sklearn.preprocessing import StandardScaler
import warnings

# Bo qua cac canh bao tu pandas
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# =============================================================================
# CÁC HÀM TIỆN ÍCH
# =============================================================================

def load_all_csv(directory):
    """Doc tat ca file CSV tu mot thu muc va gop chung lai."""
    all_files = glob.glob(os.path.join(directory, "*.csv"))
    if not all_files:
        print(f"Loi: Khong tim thay file CSV nao trong thu muc: {directory}")
        return pd.DataFrame()
    
    print(f"Tim thay {len(all_files)} file CSV. Dang doc...")
    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except pd.errors.EmptyDataError:
            print(f"Loi khi doc file {f}: File rong (No columns to parse from file)")
        except Exception as e:
            print(f"Loi khi doc file {f}: {e}")
            
    if not df_list:
        print("Loi: Khong co du lieu nao duoc doc thanh cong.")
        return pd.DataFrame()
        
    full_df = pd.concat(df_list, ignore_index=True)
    return full_df

def clean_data(df):
    """Don dep du lieu, loai bo cac cot khong can thiet va gia tri NaN."""
    # Cac cot GEE thua
    cols_to_drop = ['system:index', '.geo']
    
    # Loai bo cac cot neu chung ton tai
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    # Loai bo bat ky hang nao con gia tri NaN (rat quan trong)
    df = df.dropna()
    return df

# =============================================================================
# HÀM CHẠY CHÍNH
# =============================================================================

def main():
    RAW_DATA_DIR = '../data/raw_exports'
    PROCESSED_DATA_DIR = '../data/processed'
    MODEL_DIR = '../models'
    
    # Tao thu muc neu chua ton tai
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Buoc 1: Tai va gop du lieu
    print(f"Bat dau qua trinh tong hop du lieu tu: {RAW_DATA_DIR}")
    full_df = load_all_csv(RAW_DATA_DIR)
    
    if full_df.empty:
        print("Dung chuong trinh vi khong co du lieu.")
        return

    print(f"Tong hop thanh cong. Tong so diem mau truoc khi don dep: {len(full_df)}")
    
    # Buoc 2: Don dep du lieu
    final_df = clean_data(full_df)
    print(f"Tong so diem mau sau khi don dep: {len(final_df)}")
    
    if final_df.empty:
        print("Dung chuong trinh vi khong con du lieu sau khi don dep (dropna).")
        return
        
    print("Phan bo du lieu (0=Khong ngap, 1=Ngap):")
    print(final_df['flood'].value_counts())

    # Buoc 3: Chuan hoa du lieu (Standardization)
    
    # === SỬA LỖI: Them 'apex_date' vao danh sach loai tru ===
    exclude_cols = ['flood', 'event_id', 'purpose', 'detail', 'apex_date']
    
    # Tu dong xac dinh cac cot dac trung
    features = [col for col in final_df.columns if col not in exclude_cols]
    
    print(f"Cac dac trung se duoc su dung de huan luyen ({len(features)} cot): {features}")
    
    if not features:
        print("Loi: Khong tim thay dac trung nao de huan luyen.")
        return
        
    # Chuan hoa du lieu
    scaler = StandardScaler()
    final_df[features] = scaler.fit_transform(final_df[features])
    
    # Luu scaler de su dung sau nay
    scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Da chuan hoa du lieu va luu scaler vao: {scaler_path}")
    
    # Buoc 4: Luu file da xu ly
    output_path = os.path.join(PROCESSED_DATA_DIR, 'training_data_scaled.csv')
    final_df.to_csv(output_path, index=False)
    print(f"Da luu du lieu da xu ly thanh cong vao: {output_path}")
    
    print("\n==========================================================")
    print("HOAN TAT! Du lieu da san sang de huan luyen (train_model.py)")
    print("==========================================================")

if __name__ == "__main__":
    main()

