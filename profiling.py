import matplotlib.pyplot as plt
import numpy as np

# 数据准备
steps = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990,
         2000, 2010, 2020, 2100, 2110, 2120, 2130, 2140, 2150, 2160,
         2170, 2180, 2190, 2360, 2370, 2380, 2390, 2400, 2410, 2420,
         2430, 2440, 2450]

# 单步耗时 (ms)
iter_ms = [33.486, 34.443, 34.295, 33.797, 34.420, 33.376, 33.692, 34.911, 35.247, 36.999,
           193000, 194000, 3127.482, 2252.895, 2617.652, 2554.890, 2213.337, 2184.664, 2195.048, 2105.105,
           2509.207, 3426.971, 3309.899, 2540.899, 2537.092, 2408.803, 2357.139, 2356.556, 2306.544, 2340.231,
           2195.958, 2200.236, 2393.201]

# 吞吐量 (K Rays/Sec)
rays_k = [122.57, 119.29, 119.89, 121.54, 119.30, 123.26, 122.21, 117.78, 116.72, 111.36,
          101.51, 48.93, 1.67, 1.93, 1.75, 1.80, 2.03, 2.06, 1.97, 2.06,
          1.86, 1.48, 1.58, 1.71, 1.76, 1.85, 1.84, 1.86, 2.21, 2.18,
          1.96, 1.96, 1.80]

# 绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# 上图：单步耗时 (对数坐标，因为跨度极大)
ax1.plot(steps, iter_ms, 'o-', color='#e74c3c', markersize=4, linewidth=1.5, label='Train Iter (ms)')
ax1.set_yscale('log')
ax1.set_ylabel('Train Iter Time (ms, log scale)', fontsize=12)
ax1.set_title('Training Performance Timeline: 4-Stage Analysis', fontsize=14, fontweight='bold')
ax1.grid(True, which="both", ls="-", alpha=0.3)
ax1.legend(loc='upper left')

# 下图：吞吐量
ax2.plot(steps, rays_k, 's-', color='#2980b9', markersize=4, linewidth=1.5, label='Train Rays / Sec (K)')
ax2.set_ylabel('Throughput (K Rays/Sec)', fontsize=12)
ax2.set_xlabel('Step', fontsize=12)
ax2.grid(True, which="both", ls="-", alpha=0.3)
ax2.legend(loc='upper right')

# 添加阶段背景色块
stages = [
    (1900, 1990, '#2ecc71', 'P1: Normal Baseline'),
    (2000, 2020, '#f39c12', 'P2: Mutation Point'),
    (2100, 2190, '#e67e22', 'P3: Oscillation'),
    (2360, 2450, '#9b59b6', 'P4: New Steady State')
]

for start, end, color, label in stages:
    for ax in [ax1, ax2]:
        ax.axvspan(start, end, alpha=0.15, color=color, label=label)

# 添加阶段注释
ax1.annotate('P1: ~35ms/step\n120K Rays/s', xy=(1950, 35), fontsize=9, 
             ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ecc71", alpha=0.8))
ax1.annotate('P2: SPIKE!\n193s/step', xy=(2000, 193000), fontsize=9, 
             ha='center', xytext=(2000, 50000), arrowprops=dict(arrowstyle="->", color="#f39c12"),
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#f39c12", alpha=0.8))
ax1.annotate('P4: ~2.5s/step\n2K Rays/s', xy=(2400, 2356), fontsize=9, 
             ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#9b59b6", alpha=0.8))

plt.tight_layout()
plt.show()