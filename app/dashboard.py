import streamlit as st
from streamlit_folium import st_folium
import folium
import requests
import pandas as pd
import altair as alt

# =============================================================================
# CAU HINH TRANG
# =============================================================================
st.set_page_config(
    page_title="Hệ thống Dự báo Ngập lụt",
    page_icon="🌊",
    layout="wide"
)

# Dia chi API backend
API_URL = "http://127.0.0.1:8000"

# =============================================================================
# KHOI TAO STATE (Trang thai)
# =============================================================================
if 'map_center' not in st.session_state:
    st.session_state.map_center = [16.047079, 108.206230] # Da Nang
if 'last_clicked' not in st.session_state:
    st.session_state.last_clicked = None
if 'probability' not in st.session_state:
    st.session_state.probability = None
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

# =============================================================================
# GIAO DIEN CHINH
# =============================================================================
st.title("🌊 Hệ thống Hỗ trợ Dự báo Ngập lụt (XGBoost)")
st.caption("Dựa trên dữ liệu GEE (Sentinel-1, GPM, SMAP, SRTM) và mô hình XGBoost")

# Chia layout thanh 2 cot
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Bản đồ Tương tác")
    st.info("Nhấp vào một vị trí trên bản đồ để bắt đầu dự đoán.")
    
    # Tao ban do Folium
    m = folium.Map(location=st.session_state.map_center, zoom_start=10)
    
    # Them marker cho vi tri da chon (neu co)
    if st.session_state.last_clicked:
        folium.Marker(
            [st.session_state.last_clicked['lat'], st.session_state.last_clicked['lng']],
            popup="Vị trí đã chọn",
            icon=folium.Icon(color="blue"),
        ).add_to(m)

    # Hien thi ban do
    map_data = st_folium(m, width='100%', height=500)

    # Xu ly su kien click
    if map_data and map_data['last_clicked']:
        clicked_point = map_data['last_clicked']
        # Chi goi API neu vi tri click thay doi
        if clicked_point != st.session_state.last_clicked:
            st.session_state.last_clicked = clicked_point
            st.session_state.probability = None # Dat lai ket qua
            st.session_state.forecast_data = None
            st.session_state.error_message = None
            
            point_data = {"lat": clicked_point['lat'], "lon": clicked_point['lng']}
            
            with st.spinner("Đang lấy dữ liệu và dự đoán... (có thể mất 10-20 giây)"):
                try:
                    # Goi API /predict
                    predict_response = requests.post(f"{API_URL}/predict", json=point_data)
                    predict_response.raise_for_status() # Bao loi neu > 400
                    st.session_state.probability = predict_response.json()['probability']
                    
                    # Goi API /forecast
                    forecast_response = requests.post(f"{API_URL}/forecast", json=point_data)
                    forecast_response.raise_for_status()
                    st.session_state.forecast_data = forecast_response.json()['forecast']
                
                except requests.exceptions.RequestException as e:
                    try:
                        detail = e.response.json().get('detail', str(e))
                        st.session_state.error_message = f"Lỗi API: {detail}"
                    except:
                         st.session_state.error_message = f"Lỗi kết nối API: {e}. Bạn đã chạy 'uvicorn api:app' chưa?"

with col2:
    st.subheader("Kết quả Dự đoán")
    
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
    
    elif st.session_state.last_clicked is None:
        st.warning("Vui lòng chọn một điểm trên bản đồ.")
        
    elif st.session_state.probability is None:
        st.info("Đang chờ kết quả...")

    else:
        lat = st.session_state.last_clicked['lat']
        lon = st.session_state.last_clicked['lng']
        prob = st.session_state.probability
        
        st.metric(
            label=f"Nguy cơ Ngập tại ({lat:.4f}, {lon:.4f})",
            value=f"{prob * 100:.2f} %",
            delta=f"{prob * 100 - 50:.2f} % so với ngưỡng 50%",
            delta_color="inverse"
        )
        
        # Ve dong ho (Gauge)
        prob_percent = prob * 100
        if prob_percent < 30:
            color = "green"
            label = "Thấp"
        elif prob_percent < 70:
            color = "orange"
            label = "Trung bình"
        else:
            color = "red"
            label = "Cao"
            
        # === PHAN SUA LOI GIAO DIEN (UI) ===
        st.markdown(f"""
        <div style="
            width: 100%; 
            background-color: #eee; 
            border-radius: 10px; 
            border: 1px solid #ccc;
            overflow: visible; /* SUA LOI: Dat 'visible' de hien thi noi dung tran */
        ">
            <div style="
                width: {prob_percent}%; 
                background-color: {color}; 
                color: white; 
                text-align: center; 
                padding: 10px 0; 
                font-weight: bold;
                transition: width 0.5s ease-in-out;
                min-width: 100px; /* Dat chieu rong toi thieu de chua text */
            ">
                {label} ({prob_percent:.1f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
        # === KET THUC SUA LOI GIAO DIEN ===
        
        # Ve bieu do du bao mua
        st.subheader("Dự báo Mưa (GFS)")
        if st.session_state.forecast_data:
            try:
                df_forecast = pd.DataFrame(st.session_state.forecast_data)
                
                if df_forecast.empty:
                    st.info("Không có dữ liệu dự báo mưa (API trả về rỗng).")
                else:
                    df_forecast['time'] = pd.to_datetime(df_forecast['time'])
                    
                    # Nhom cac gia tri 3-gio lai thanh tong-cong-moi-ngay
                    df_daily = df_forecast.set_index('time').resample('D').sum(numeric_only=True).reset_index()
                    df_daily = df_daily.rename(columns={'precipitation_mm_3hr': 'precipitation_mm_daily'})

                    chart = alt.Chart(df_daily).mark_bar().encode(
                        x=alt.X('time:T', title='Ngày', axis=alt.Axis(format="%Y-%m-%d")),
                        y=alt.Y('precipitation_mm_daily:Q', title='Lượng mưa (mm/ngày)'),
                        tooltip=[
                            alt.Tooltip('time:T', title='Ngày', format="%Y-%m-%d"), 
                            alt.Tooltip('precipitation_mm_daily:Q', title='Lượng mưa (mm)', format=".1f")
                        ]
                    ).properties(
                        title="Tổng lượng mưa dự báo hàng ngày"
                    ).interactive()
                    
                    st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"Lỗi khi vẽ biểu đồ dự báo: {e}")
        else:
            st.info("Không có dữ liệu dự báo mưa.")

