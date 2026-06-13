import matplotlib.pyplot as plt
import numpy as np

ratios = [10, 20, 30, 40, 50]

bapnp  = [93.5, 93.5, 93.0, 93.0, 93.0]
epnp   = [93.5, 93.0, 94.0, 94.0, 92.5]
cpnp   = [93.0, 93.0, 94.0, 93.5, 84.1]
sqpnp  = [94.5, 94.0, 93.5, 93.5, 93.5]

plt.figure(figsize=(7, 4.5))
plt.plot(ratios, bapnp,  'ro-',  linewidth=2, markersize=8, label='BAPnP+RANSAC')
plt.plot(ratios, epnp,   'bs--', linewidth=2, markersize=8, label='EPnP+RANSAC')
plt.plot(ratios, cpnp,   'g^-.', linewidth=2, markersize=8, label='CPnP+RANSAC')
plt.plot(ratios, sqpnp,  'mv:',  linewidth=2, markersize=8, label='SQPnP+RANSAC')

plt.xlabel('Injected Extremal Outlier Ratio (%)', fontsize=12)
plt.ylabel('Success Rate (%)', fontsize=12)
plt.ylim(75, 100)
plt.xticks(ratios)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower left', fontsize=10)

plt.tight_layout()
plt.savefig(r'D:\研究生学习\pnp\pnp\论文写作 - 副本\final\review2_outlier_succ.pdf',
            bbox_inches='tight', pad_inches=0.05)
plt.savefig(r'D:\研究生学习\pnp\pnp\论文写作 - 副本\final\review2_outlier_succ.png',
            bbox_inches='tight', pad_inches=0.05, dpi=200)
print('Done.')
