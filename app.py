import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

st.set_page_config(page_title="AI地形要素自动识别 Demo", layout="wide")

st.title("AI自动识别关键地形要素系统")
st.write("上传地图或遥感影像，系统自动识别河流、道路、山脊等关键地形要素")

uploaded = st.file_uploader("上传地图影像", type=["jpg","png","jpeg"])

show_river = st.sidebar.checkbox("显示河流", True)
show_road = st.sidebar.checkbox("显示道路", True)
show_ridge = st.sidebar.checkbox("显示山脊", True)
show_points = st.sidebar.checkbox("显示关键点", True)

if uploaded:

    image = Image.open(uploaded)
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    st.subheader("原始地图")
    st.image(image,use_column_width=True)

    st.subheader("AI自动识别分析")

    # -------- 河流识别 --------
    lower_blue = np.array([0,0,100])
    upper_blue = np.array([120,120,255])
    river_mask = cv2.inRange(img,lower_blue,upper_blue)

    contours,_ = cv2.findContours(
        river_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    river_lines = []

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            river_lines.append(cnt)

    # -------- 山脊检测 --------
    edges = canny(gray, sigma=2)

    ridge_lines = probabilistic_hough_line(
        edges,
        threshold=10,
        line_length=80,
        line_gap=10
    )

    # -------- 道路检测 --------
    road_edges = cv2.Canny(gray,50,150)

    lines = cv2.HoughLinesP(
        road_edges,
        1,
        np.pi/180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )

    road_lines = []

    if lines is not None:
        for l in lines:
            x1,y1,x2,y2 = l[0]
            road_lines.append((x1,y1,x2,y2))

    # -------- 关键点生成 --------
    h,w = gray.shape

    key_points = [
        (w*0.3,h*0.3),
        (w*0.6,h*0.5),
        (w*0.8,h*0.2)
    ]

    # -------- 绘图 --------

    fig = go.Figure()

    fig.add_trace(go.Image(z=img))

    # 河流
    if show_river:
        for cnt in river_lines:
            x = cnt[:,0][:,0]
            y = cnt[:,0][:,1]

            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color="blue",width=3),
                name="河流"
            ))

    # 山脊
    if show_ridge:
        for line in ridge_lines:
            p0,p1 = line

            fig.add_trace(go.Scatter(
                x=[p0[0],p1[0]],
                y=[p0[1],p1[1]],
                mode="lines",
                line=dict(color="red",width=3),
                name="山脊"
            ))

    # 道路
    if show_road:
        for r in road_lines:

            fig.add_trace(go.Scatter(
                x=[r[0],r[2]],
                y=[r[1],r[3]],
                mode="lines",
                line=dict(color="yellow",width=2),
                name="道路"
            ))

    # 关键点
    if show_points:

        xs = [p[0] for p in key_points]
        ys = [p[1] for p in key_points]

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=12,color="red"),
            name="关键点"
        ))

    fig.update_layout(
        width=900,
        height=700,
        margin=dict(l=0,r=0,t=0,b=0)
    )

    st.plotly_chart(fig)

    # -------- 统计 --------

    st.subheader("识别统计")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("河流数量",len(river_lines))
    col2.metric("道路数量",len(road_lines))
    col3.metric("山脊数量",len(ridge_lines))
    col4.metric("关键点数量",len(key_points))

else:

    st.info("请上传地图影像开始识别")
