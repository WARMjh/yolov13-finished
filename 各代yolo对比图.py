import matplotlib.pyplot as plt

# 提取并修正后的精确数据 (FLOPs, mAP)
# 注意：YOLOv8 新增了第四个点 (257.8, 53.9) 以还原原图中向右侧延伸出界的折线
data = {
    'YOLOv6-3.0': {'x': [12, 45, 150], 'y': [37.0, 44.2, 51.8], 'color': '#55a624', 'marker': 's', 'linestyle': '--'},
    'Gold-YOLO':  {'x': [15, 45, 150], 'y': [39.5, 45.4, 51.8], 'color': '#d39200', 'marker': 'o', 'linestyle': '--'},
    'YOLOv8':     {'x': [8.7, 28.4, 165.2, 257.8], 'y': [37.3, 44.9, 52.9, 53.9], 'color': '#ff7f0e', 'marker': '^', 'linestyle': '--'},
    'YOLOv9':     {'x': [25, 102], 'y': [46.6, 53.0], 'color': '#7f7f00', 'marker': 'v', 'linestyle': '--'},
    'YOLOv10':    {'x': [6.7, 21.6, 120, 160.4], 'y': [38.5, 46.3, 53.2, 54.4], 'color': '#704238', 'marker': 'D', 'linestyle': '--'},
    'YOLOv11':    {'x': [6, 20, 85, 195], 'y': [38.5, 45.8, 52.2, 54.2], 'color': '#00c3c3', 'marker': '<', 'linestyle': '--'},
    'YOLOv12':    {'x': [6, 20, 85, 200], 'y': [40.0, 47.0, 53.0, 54.4], 'color': 'blue', 'marker': '>', 'linestyle': '--'},
    'YOLOv13':    {'x': [6, 20, 85, 198], 'y': [41.6, 48.0, 53.4, 54.8], 'color': '#ff1a33', 'marker': '*', 'linestyle': '-'}
}

# 实例化画布对象并设置尺寸
fig, ax = plt.subplots(figsize=(9, 7))

# 遍历字典，绘制包含标记和连线的折线系
for name, props in data.items():
    # 针对基线模型 YOLOv13 进行视觉强化（加粗实线与大号星标）
    if name == 'YOLOv13':
        linewidth = 2.5
        markersize = 13
        zorder = 10  # 确保主图层位于最上方
    else:
        linewidth = 1.8
        markersize = 8
        zorder = 5

    ax.plot(props['x'], props['y'], label=name, color=props['color'],
             marker=props['marker'], linestyle=props['linestyle'],
             linewidth=linewidth, markersize=markersize, zorder=zorder)

# 设定衬线体(Serif)字体，高度还原学术论文标准图表规范
font_kwargs = {'fontfamily': 'serif', 'fontweight': 'bold'}

ax.set_xlabel('FLOPs (G)', fontsize=15, **font_kwargs)
ax.set_ylabel('MS COCO mAP (%)', fontsize=15, **font_kwargs)

# 锁定视图边界（YOLOv8 的最后一个点在257.8，将被此边界自然截断）
ax.set_xlim(-10, 210)
ax.set_ylim(36.5, 55.5)

# 配置刻度标签属性
plt.xticks(fontsize=12, fontfamily='serif')
plt.yticks(fontsize=12, fontfamily='serif')

# 配置图例（白色背景、黑色边框）
legend = ax.legend(loc='lower right', fontsize=12, frameon=True, edgecolor='black')
for text in legend.get_texts():
    text.set_fontfamily('serif')
    text.set_fontweight('bold')

# 强化图表外围边框（学术风无网格、粗边框）
for spine in ax.spines.values():
    spine.set_linewidth(1.8)
    spine.set_color('black')

# 执行渲染与紧凑布局
plt.tight_layout()
plt.show()