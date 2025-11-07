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
st.title("🌊 Hệ thống Hỗ trợ Dự báo Ngập lụt")
st.caption("Dự báo ngập lụt dựa trên mô hình XGBoost và dữ liệu vệ tinh:")
st.caption("- Địa hình: SRTM")
st.caption("- Lớp phủ: ESA WorldCover")
st.caption("- Lượng mưa quá khứ: GPM IMERG")
st.caption("- Dự báo mưa: NOAA GFS")
st.caption("- Độ ẩm đất: NASA SMAP")

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
                    predict_data = predict_response.json()
                    st.session_state.probability = predict_data['probability']
                    st.session_state.features = predict_data.get('features', {})
                    
                    # Goi API /forecast
                    forecast_response = requests.post(f"{API_URL}/forecast", json=point_data)
                    forecast_response.raise_for_status()
                    forecast_json = forecast_response.json()
                    
                    # DEBUG: Hiển thị response từ API
                    st.sidebar.write("### DEBUG: API Response")
                    st.sidebar.json(forecast_json)
                    
                    st.session_state.forecast_data = forecast_json  # Lưu toàn bộ response
                
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
        st.info("👆 Vui lòng chọn một điểm trên bản đồ để xem dự báo.")
    
    else:
        # Tab cho các loại dự báo khác nhau
        tab1, tab2 = st.tabs(["🌡️ Dự báo hiện tại", "📅 Dự báo 7 ngày"])
        
        with tab1:
            # Hiển thị dự báo hiện tại
        st.warning("Vui lòng chọn một điểm trên bản đồ.")
        
    elif st.session_state.probability is not None:
        st.write("### 1️⃣ Dự báo hiện tại")
        # Hiển thị xác suất ngập hiện tại
        prob = st.session_state.probability * 100
        if prob > 70:
            st.error(f"⚠️ Nguy cơ ngập cao: {prob:.1f}%")
        elif prob > 30:
            st.warning(f"⚠️ Nguy cơ ngập trung bình: {prob:.1f}%")
        else:
            st.success(f"✓ Nguy cơ ngập thấp: {prob:.1f}%")
            
        # Hiển thị dự báo cho 7 ngày tới
        if st.session_state.forecast_data:
            st.write("### 2️⃣ Dự báo 7 ngày tới")
            forecast_data = st.session_state.forecast_data['forecast']
            
            for day_data in forecast_data:
                date = pd.to_datetime(day_data['date']).strftime("%d/%m/%Y")
                prob = day_data['flood_probability'] * 100
                rain = day_data['precipitation_mm_24hr']
                
                # Tạo styled container cho mỗi ngày
                if prob > 70:
                    color = "red"
                    emoji = "🔴"
                elif prob > 30:
                    color = "orange"
                    emoji = "🟡"
                else:
                    color = "green"
                    emoji = "🟢"
                    
                st.markdown(f"""
                <div style="padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 5px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{date}</strong> {emoji}
                        </div>
                        <div>
                            🌧️ {rain:.1f}mm &nbsp;|&nbsp; 
                            <span style="color: {color}">⚠️ {prob:.1f}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Vẽ biểu đồ lượng mưa chi tiết
            st.write("### 3️⃣ Dự báo mưa chi tiết (3h)")
            if 'detail' in st.session_state.forecast_data and 'forecast_3h' in st.session_state.forecast_data['detail']:
                df_detail = pd.DataFrame(st.session_state.forecast_data['detail']['forecast_3h'])
                df_detail['time'] = pd.to_datetime(df_detail['time'])
                
                chart = alt.Chart(df_detail).mark_bar().encode(
                    x=alt.X('time:T', 
                          title='Thời gian',
                          axis=alt.Axis(format="%d/%m %H:00", labelAngle=-45)),
                    y=alt.Y('precipitation_mm_3hr:Q', 
                          title='Lượng mưa (mm/3h)'),
                    tooltip=[
                        alt.Tooltip('time:T', title='Thời gian', format="%Y-%m-%d %H:00"), 
                        alt.Tooltip('precipitation_mm_3hr:Q', title='Lượng mưa (mm/3h)', format=".1f")
                    ]
                ).properties(
                    title="Lượng mưa dự báo chi tiết (3 giờ một lần)",
                    height=300
                ).interactive()
                
                st.altair_chart(chart, use_container_width=True)
        
        # Hiển thị chi tiết các đặc trưng
        with st.expander("Chi tiết đặc điểm địa hình và khu vực"):
            features = st.session_state.features
            if features:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Đặc điểm địa hình")
                    if 'elevation' in features:
                        st.write(f"🏔️ Độ cao: {features['elevation']:.1f}m")
                    if 'slope' in features:
                        st.write(f"📐 Độ dốc: {features['slope']:.1f}°")
                    
                    st.subheader("Lượng mưa tích lũy")
                    if 'precip_3_day' in features:
                        st.write(f"🌧️ 3 ngày: {features['precip_3_day']:.1f}mm")
                    if 'precip_7_day' in features:
                        st.write(f"🌧️ 7 ngày: {features['precip_7_day']:.1f}mm")
                    if 'precip_14_day' in features:
                        st.write(f"🌧️ 14 ngày: {features['precip_14_day']:.1f}mm")
                
                with col2:
                    st.subheader("Phân tích khu vực")
                    flags = {
                        'is_flood_prone': ('🌊 Vùng dễ ngập', 'Vùng ít ngập'),
                        'is_permanent_water': ('💧 Vùng nước', 'Vùng khô'),
                        'is_urban': ('🏘️ Khu dân cư', 'Không phải khu dân cư'),
                        'is_agriculture': ('🌾 Đất nông nghiệp', 'Không phải đất nông nghiệp')
                    }
                    
                    for flag, (true_text, false_text) in flags.items():
                        if flag in features:
                            if features[flag] > 0:
                                st.info(true_text)
                            else:
                                st.write(false_text)
        
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
        
        # === PHAN HIEN THI DU BAO ===
        st.subheader("🌧️ Dự báo Mưa và Ngập lụt (7 ngày tới)")
        if st.session_state.forecast_data:
            try:
                df_forecast = pd.DataFrame(st.session_state.forecast_data)
                
                if df_forecast.empty:
                    st.info("Không có dữ liệu dự báo (API trả về rỗng).")
                else:
                    # 1. BANG DU BAO TONG HOP
                    st.write("### Dự báo theo ngày")
                    
                    for idx, row in df_forecast.iterrows():
                        date = pd.to_datetime(row['date']).strftime("%d/%m/%Y")
                        prob = row['flood_probability'] * 100
                        rain = row['precipitation_mm_24hr']
                        
                        # Tao styled container cho moi ngay
                        color = "red" if prob > 70 else "orange" if prob > 30 else "green"
                        warning = "CAO" if prob > 70 else "TRUNG BÌNH" if prob > 30 else "THẤP"
                        
                        st.markdown(f"""
                        <div style="padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 5px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong>{date}</strong>
                                </div>
                                <div>
                                    🌧️ {rain:.1f}mm/24h &nbsp;|&nbsp; 
                                    <span style="color: {color}">⚠️ {prob:.1f}% ({warning})</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # 2. BIEU DO LUONG MUA CHI TIET
                    st.write("### Dự báo mưa chi tiết (3 giờ một lần)")
                    
                    if 'detail' in st.session_state and 'forecast_3h' in st.session_state['detail']:
                        df_detail = pd.DataFrame(st.session_state['detail']['forecast_3h'])
                        df_detail['time'] = pd.to_datetime(df_detail['time'])
                        
                        detail_chart = alt.Chart(df_detail).mark_bar().encode(
                            x=alt.X('time:T', 
                                  title='Thời gian',
                                  axis=alt.Axis(format="%d/%m %H:00", labelAngle=-45)),
                            y=alt.Y('precipitation_mm_3hr:Q', 
                                  title='Lượng mưa (mm/3h)'),
                            tooltip=[
                                alt.Tooltip('time:T', title='Thời gian', format="%Y-%m-%d %H:00"), 
                                alt.Tooltip('precipitation_mm_3hr:Q', title='Lượng mưa (mm/3h)', format=".1f")
                            ],
                            color=alt.value("#5B9BD5")  # Màu xanh dương nhạt
                        ).properties(
                            title="Lượng mưa dự báo chi tiết (3 giờ một lần)",
                            height=250
                        ).interactive()
                        
                        st.altair_chart(detail_chart, use_container_width=True)
                    
                    # 3. BIEU DO NGUY CO NGAP
                    st.write("### Diễn biến nguy cơ ngập")
                    
                    risk_chart = alt.Chart(df_forecast).mark_line(point=True).encode(
                        x=alt.X('date:T', 
                              title='Ngày',
                              axis=alt.Axis(format="%d/%m", labelAngle=0)),
                        y=alt.Y('flood_probability:Q', 
                              title='Nguy cơ ngập (%)',
                              scale=alt.Scale(domain=[0, 1])),
                        tooltip=[
                            alt.Tooltip('date:T', title='Ngày', format="%Y-%m-%d"), 
                            alt.Tooltip('flood_probability:Q', title='Nguy cơ ngập', format=".1%"),
                            alt.Tooltip('precipitation_mm_24hr:Q', title='Lượng mưa (mm/24h)', format=".1f")
                        ],
                        color=alt.value("#FF7F7F")  # Màu đỏ nhạt
                    ).properties(
                        title="Diễn biến nguy cơ ngập trong 7 ngày tới",
                        height=250
                    ).interactive()
                    
                    # Thêm đường ngưỡng cảnh báo
                    warning_rule = alt.Chart(pd.DataFrame({
                        'y': [0.3, 0.7],
                        'level': ['Ngưỡng cảnh báo thấp', 'Ngưỡng cảnh báo cao']
                    })).mark_rule(strokeDash=[5, 5]).encode(
                        y='y:Q',
                        color=alt.Color('level:N', 
                                      scale=alt.Scale(domain=['Ngưỡng cảnh báo thấp', 'Ngưỡng cảnh báo cao'],
                                                    range=['orange', 'red'])),
                        size=alt.value(1)
                    )
                    
                    st.altair_chart(risk_chart + warning_rule, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Lỗi khi hiển thị dự báo: {e}")
                import traceback
                st.error(traceback.format_exc())
        else:
            st.info("Không có dữ liệu dự báo.")

