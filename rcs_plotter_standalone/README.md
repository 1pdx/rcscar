# RCS 数据绘图工具

这是从主程序中单独拆出的 RCS 表格绘图工具，只依赖本目录内文件运行。

## 功能

- 导入 Cluster RCS `CSV/XLSX/XLSM` 表格。
- 生成直线距离-RCS 曲线，可叠加多文件对比，并按主程序同样的分箱融合方式生成一条 `combined` 曲线。
- 生成圆周 RCS 极坐标折线图，可导入 `__orbit_rcs__` 圆周表格，也可把 Cluster Raw 按时间平铺为 0-360° 折线图。
- 生成空间点云图：导入 Cluster Raw 后，每帧以 `DX00/DY00` 为主簇中心，展开 `DX00..DX19 / DY00..DY19 / RCS00..RCS19` 所有有效簇，绘制相对目标中心的 `Y-X` 点云，颜色为 `RCS + 标定值`。
- 选择参考产品和角度，直线图可叠加参考上限/下限。
- 支持 RCS 标定值偏移和保存图片。
- 保存直线距离-RCS 图片时，同步生成 `*_summary.xlsx` 汇总文件；汇总表按 0.1m 分箱且只保留 50m 以内，RCS1/RCS2/... 取自图中各组曲线，RCSAVA 取自图中 combined 曲线，全部叠加同一个标定值，并包含保存图片和参考上下限表。

## 运行

直接运行：

```bat
dist\RCS_Data_Plotter.exe
```

## 重新打包

```bat
build_exe.bat
```
