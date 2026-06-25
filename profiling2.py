import matplotlib.pyplot as plt
import numpy as np

# 数据准备
strategies = ['P4 Baseline', 'Vec + Cache', 'Skip Iter', 'Drop Batch']
step_times = [2.50, 1.94, 1.31, 1.75]          # 单位: 秒
throughputs = [1.85, 2.38, 3.53, 2.64]         # 单位: K Rays/Sec

# 学术配色 (无底色)
colors_step = ['#404040', '#66c2a5', '#fc8d62', '#8da0cb']
colors_tp   = ['#404040', '#74add1', '#f46d43', '#abd9e9']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# --- 左图: Step Time ---
bars1 = ax1.bar(strategies, step_times, color=colors_step, edgecolor='black', linewidth=0.8, width=0.6)
for bar, val in zip(bars1, step_times):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
             f'{val:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.set_ylabel('Step Time (s)', fontsize=13)
ax1.set_title('Estimated Step Time', fontsize=14, fontweight='bold', pad=12)
ax1.set_ylim(0, 3.2)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='both', labelsize=11)

# --- 右图: Throughput ---
bars2 = ax2.bar(strategies, throughputs, color=colors_tp, edgecolor='black', linewidth=0.8, width=0.6)
for bar, val in zip(bars2, throughputs):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06,
             f'{val:.2f}K', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.set_ylabel('Throughput (K Rays/Sec)', fontsize=13)
ax2.set_title('Estimated Throughput', fontsize=14, fontweight='bold', pad=12)
ax2.set_ylim(0, 4.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(axis='both', labelsize=11)

# 统一英文 Caption
fig.text(0.5, -0.02, 
         'Figure 1. Comparison of estimated step time and throughput under P4 baseline and acceleration strategies.',
         ha='center', fontsize=12, style='italic')

plt.tight_layout()
plt.show()