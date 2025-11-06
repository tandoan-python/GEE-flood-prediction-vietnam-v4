import pandas as pd
import os
import glob
# import joblib -> ĐÃ XÓA (Khong chuan hoa o day)
# from sklearn.preprocessing import StandardScaler -> ĐÃ XÓA

# Dinh nghia cac duong dan
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'raw_exports'))
PROCESSED_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'processed'))
# MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models')) # XOA

# Tao thu muc neu chua ton tai
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
# os.makedirs(MODEL_DIR, exist_ok=True) # XOA

def main():
    print(f"Bat dau qua trinh tong hop du lieu tu: {RAW_DATA_DIR}")
    
    csv_files = glob.glob(os.path.join(RAW_DATA_DIR, '*.csv'))
    
    if not csv_files:
        print(f"Loi: Khong tim thay file CSV nao trong {RAW_DATA_DIR}.")
        print("Vui long kiem tra lai: Ban da tai cac file CSV tu GEE ve dung thu muc chua?")
        return

    print(f"Tim thay {len(csv_files)} file CSV. Dang doc...")
    
    all_dataframes = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                print(f"Canh bao: File {os.path.basename(f)} bi rong (empty). Bo qua.")
                continue
            all_dataframes.append(df)
        except pd.errors.EmptyDataError:
            print(f"Loi khi doc file {os.path.basename(f)}: File rong (No columns to parse from file)")
        except Exception as e:
            print(f"Loi khi doc file {os.path.basename(f)}: {e}")

    if not all_dataframes:
        print("Loi: Khong co du lieu hop le de tong hop. Dung chuong trinh.")
        return

    final_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Tong hop thanh cong. Tong so diem mau truoc khi don dep: {len(final_df)}")

    # Buoc 1: Don dep du lieu
    # Loai bo cac cot khong can thiet
    # .geo va system:index la metadata cua GEE
    cols_to_drop = [col for col in final_df.columns if col.startswith('.geo') or col == 'system:index']
    final_df = final_df.drop(columns=cols_to_drop)
    
    # Kiem tra va loai bo NaN (rat quan trong)
    # NaN co the xay ra do SMAP (soil moisture) bi loi
    final_df = final_df.dropna()
    print(f"Tong so diem mau sau khi don dep (loai bo NaN): {len(final_df)}")

    # Buoc 2: Hien thi phan bo du lieu
    if 'flood' in final_df.columns:
        print("\nPhan bo du lieu (0=Khong ngap, 1=Ngap):")
        print(final_df['flood'].value_counts())
    else:
        print("Canh bao: Khong tim thay cot 'flood'.")
        
    # Buoc 3: Luu file DU LIEU THO (chua chuan hoa)
    output_path = os.path.join(PROCESSED_DATA_DIR, 'combined_data_raw.csv')
    final_df.to_csv(output_path, index=False)
    
    print(f"\n==================================================================")
    print(f"HOAN TAT! Da luu du lieu THO thanh cong vao: {output_path}")
    print(f"BUOC TIEP THEO: Chay 'python src/train_model.py' de huan luyen mo hinh.")
    print(f"==================================================================")

if __name__ == "__main__":
    main()

