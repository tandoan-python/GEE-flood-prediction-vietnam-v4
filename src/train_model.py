import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import optuna
import shap
import matplotlib.pyplot as plt
import os
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler

# =============================================================================
# ĐỊNH NGHĨA ĐƯỜNG DẪN
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'processed'))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models'))
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'outputs'))

# Tao thu muc neu chua ton tai
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dinh nghia cac file
DATA_PATH = os.path.join(DATA_DIR, 'combined_data_raw.csv') 
MODEL_PATH = os.path.join(MODEL_DIR, 'flood_model.xgb')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib') 
REPORT_PATH = os.path.join(OUTPUT_DIR, 'model_evaluation_report.txt')
SHAP_PLOT_PATH = os.path.join(OUTPUT_DIR, 'shap_summary_plot.png')

# =============================================================================
# CÁC HÀM TIỆN ÍCH
# =============================================================================

def save_report(content):
    """Luu bao cao van ban vao file."""
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\nDa luu bao cao danh gia chi tiet vao: {REPORT_PATH}")

def plot_shap_summary(shap_values, features, feature_names):
    """Ve va luu do thi SHAP summary."""
    try:
        plt.figure()
        shap.summary_plot(shap_values, features, feature_names=feature_names, show=False)
        plt.savefig(SHAP_PLOT_PATH, bbox_inches='tight')
        plt.close()
        print(f"Da luu do thi SHAP summary vao: {SHAP_PLOT_PATH}")
    except Exception as e:
        print(f"Loi khi ve SHAP plot: {e}. Co the do X_train bi rong.")


# =============================================================================
# HÀM HUẤN LUYỆN CHÍNH
# =============================================================================

def main():
    print("Bat dau qua trinh huan luyen mo hinh...")
    
    # --- BUOC 1: DOC VA PHAN CHIA DU LIEU ---
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Loi: Khong tim thay file du lieu tai: {DATA_PATH}")
        print("Vui long chay 'combine_data.py' truoc.")
        return

    print(f"Da doc thanh cong {len(df)} dong du lieu tu {DATA_PATH}")

    features = [col for col in df.columns if col not in [
        'flood', 'event_id', 'purpose', 'apex_date', 'detail', 's1_diff', '.geo', 'system:index'
    ]]
    
    if not features:
        print("Loi: Khong tim thay dac trung nao de huan luyen.")
        return
        
    print(f"Su dung {len(features)} dac trung de huan luyen: {features}")
    target = 'flood'

    try:
        train_df = df[df['purpose'] == 'training'].copy()
        val_df = df[df['purpose'] == 'validation'].copy()
        test_df = df[df['purpose'] == 'testing'].copy()
    except Exception as e:
        print(f"Loi khi phan chia du lieu: {e}")
        print("Vui long kiem tra lai cot 'purpose' trong file CSV.")
        return

    if train_df.empty or val_df.empty or test_df.empty:
        print("Loi: Mot trong cac tap (training, validation, testing) bi rong.")
        print(f"Training: {len(train_df)}, Validation: {len(val_df)}, Testing: {len(test_df)}")
        print("Vui long kiem tra lai file 'FLOOD_EVENTS' trong 'prepare_data.py'.")
        return
        
    print(f"Phan chia du lieu thanh cong:")
    print(f"- Tap Training: {len(train_df)} diem")
    print(f"- Tap Validation: {len(val_df)} diem")
    print(f"- Tap Testing: {len(test_df)} diem")

    # --- BUOC 2: CHUAN HOA DU LIEU (SCALING) ---
    print("\nBat dau chuan hoa du lieu (StandardScaler)...")
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(train_df[features])
    y_train = train_df[target]
    
    X_val = scaler.transform(val_df[features])
    y_val = val_df[target]
    
    X_test = scaler.transform(test_df[features])
    y_test = test_df[target]
    
    print("Chuan hoa du lieu thanh cong.")
    joblib.dump(scaler, SCALER_PATH)
    print(f"Da luu scaler (da fit) vao: {SCALER_PATH}")

    # --- BUOC 3: XU LY MAT CAN BANG DU LIEU (CHO TAP TRAINING) ---
    print("\nXu ly mat can bang du lieu (tinh toan 'sample_weight')...")
    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )
    print(f"Phan bo nhan 'flood' tap training:\n{y_train.value_counts()}")

    # --- BUOC 4: TIM KIEM SIEU THAM SO (OPTUNA) ---
    print("\nBat dau tim kiem sieu tham so voi Optuna...")
    
    def objective(trial):
        # === SUA LOI O DAY ===
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss', 
            'n_estimators': 1000, 
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50 # early_stopping_rounds PHAI dat o day
        }
        # === KET THUC SUA ===
        
        model = xgb.XGBClassifier(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)], # Tham so nay PHAI co mat
            # early_stopping_rounds va eval_metric KHONG dat o day
            verbose=False,
            sample_weight=sample_weights 
        )
        
        report_dict = classification_report(y_val, model.predict(X_val), output_dict=True)
        f1_score_class_1 = report_dict.get('1', {}).get('f1-score', 0)
        return f1_score_class_1

    study = optuna.create_study(direction='maximize')
    try:
        study.optimize(objective, n_trials=50) 
        print(f"Tim kiem hoan tat. Best F1-score (class 1, tren tap validation): {study.best_value:.4f}")
        print(f"Tham so tot nhat: {study.best_params}")
        best_params = study.best_params
    except Exception as e:
        print(f"Loi trong qua trinh Optuna: {e}")
        print("Su dung tham so mac dinh de tiep tuc.")
        best_params = {} 

    # --- BUOC 5: HUAN LUYEN MO HINH CUOI CUNG ---
    print("\nHuan luyen mo hinh cuoi cung voi tham so tot nhat...")
    
    # === SUA LOI O DAY ===
    # Them cac tham so co dinh vao best_params
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'logloss' 
    best_params['n_estimators'] = 1000 
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['early_stopping_rounds'] = 50 # early_stopping_rounds PHAI dat o day
    # === KET THUC SUA ===

    final_model = xgb.XGBClassifier(**best_params)
    
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)], # Tham so nay PHAI co mat
        # early_stopping_rounds va eval_metric KHONG dat o day
        verbose=False,
        sample_weight=sample_weights
    )
    
    joblib.dump(final_model, MODEL_PATH)
    print(f"Da luu mo hinh vao: {MODEL_PATH}")

    # --- BUOC 6: DANH GIA MO HINH TREN TAP TEST ---
    print("\nBat dau danh gia mo hinh tren tap TEST (du lieu chua tung thay)...")
    
    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)[:, 1] 

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['0_KhongNgap', '1_Ngap'])
    cm = confusion_matrix(y_test, y_pred)
    
    report_content = f"""
    ==================================================================
    BAO CAO DANH GIA MO HINH (TREN TAP TEST)
    ==================================================================
    
    Tong quan:
    - So diem tap Test: {len(y_test)}
    - Phan bo nhan tap Test:
    {y_test.value_counts().to_string()}
    
    Do chinh xac tong a (Accuracy): {accuracy:.4f}
    
    Ma tran nham lan (Confusion Matrix):
    {cm}
    
    Chi tiet (Precision, Recall, F1-Score):
    {report}
    
    Cac tham so da su dung:
    {best_params}
    
    ==================================================================
    """
    
    print(report_content)
    save_report(report_content)

    # --- BUOC 7: GIAI THICH MO HINH (SHAP) ---
    print("\nBat dau tinh toan gia tri SHAP (co the mat vai phut)...")
    try:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_train)
        
        plot_shap_summary(shap_values, X_train, features)
    except Exception as e:
        print(f"Loi khi tinh toan hoac ve SHAP: {e}")
    
    print(f"\n==================================================================")
    print(f"HOAN TAT! Da huan luyen, danh gia va luu mo hinh.")
    print(f"File mo hinh: {MODEL_PATH}")
    print(f"File scaler: {SCALER_PATH}")
    print(f"File bao cao: {REPORT_PATH}")
    print(f"File SHAP plot: {SHAP_PLOT_PATH}")
    print(f"==================================================================")

if __name__ == "__main__":
    main()

