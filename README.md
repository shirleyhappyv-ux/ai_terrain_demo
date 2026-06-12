# ⛰️ 重庆平行岭谷郊外山体坡度与比高分析引擎 (C++ & GDAL)

本项目基于 **C++ 11** 与 **GDAL (Geospatial Data Abstraction Library)** 开发，部署于 Codespace 在线平台。用于针对中大型（700MB+）高程数据（DEM）进行秒级核心地形特征提取、空间分析及反向坡度频数直方图统计。

## 🌟 核心产品特性
1. **高性能分块 I/O 测算**：基于 GDAL 的 `RasterIO` 窗口读取技术，面对 40930 x 33522 的超大像素 TIF 栅格图形，内存开销控制在 50MB 以内，计算响应在毫秒级。
2. **地理坐标系自适应纠偏**：在底层自动识别并拦截 WGS84 经纬度（度）与海拔（米）的水平/垂直单位畸变，动态引入北纬 30° 投影变换系数，彻底规避传统 GIS 算法中坡度逼近 90° 的工程假象。
3. **反向坡度频数矩阵**：支持以区域平均坡度为基准，以 5 度为步长，全自动产出高精度的空间地形占比结构报表，为户外活动、无人机避障与滑坡灾害监控提供结构化决策支持。

## 🚀 快速开始

### 1. 环境依赖 (Ubuntu/Codespace)
```bash
sudo apt-get update && sudo apt-get install -y libgdal-dev gdal-bin g++

### 2. 编译 (Ubuntu/Codespace)
```bash
g++ -O3 mountain_analysis.cpp $(gdal-config --cflags) -o mountain_analysis -lgdal

### 2. 运行 (Ubuntu/Codespace)
```bash
./mountain_analysis
