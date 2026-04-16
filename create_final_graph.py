import matplotlib.pyplot as plt
import numpy as np

# =============================================
# GRAPH 1: BUFFER OPTIMIZATION (Accuracy vs Latency)
# =============================================
plt.style.use('default')
fig, ax1 = plt.subplots(figsize=(10, 6))

# Your actual data
buffer_sizes = [6, 8, 10, 12, 14, 16, 18, 20]
accuracy = [86.0, 89.0, 92.0, 95.0, 95.5, 96.0, 96.5, 97.0]
latency = [0.17, 0.27, 0.33, 0.40, 0.53, 0.60, 0.67, 0.67]

# Plot accuracy on left y-axis
color = 'royalblue'
ax1.set_xlabel('Buffer Size (frames)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', color=color, fontsize=12, fontweight='bold')
line1 = ax1.plot(buffer_sizes, accuracy, 'o-', color=color, linewidth=2.5, 
                  markersize=8, markerfacecolor='white', markeredgewidth=2, 
                  label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([80, 100])
ax1.grid(True, alpha=0.3, linestyle='--')

# Plot latency on right y-axis
ax2 = ax1.twinx()
color = 'darkorange'
ax2.set_ylabel('Latency (seconds)', color=color, fontsize=12, fontweight='bold')
line2 = ax2.plot(buffer_sizes, latency, 's-', color=color, linewidth=2.5,
                  markersize=8, markerfacecolor='white', markeredgewidth=2,
                  label='Latency')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0, 0.8])

# Add value labels
for i, (bs, acc, lat) in enumerate(zip(buffer_sizes, accuracy, latency)):
    ax1.annotate(f'{acc:.0f}%', (bs, acc), 
                xytext=(0, 8), textcoords='offset points', 
                ha='center', fontsize=9, fontweight='bold')
    ax2.annotate(f'{lat:.2f}s', (bs, lat), 
                xytext=(0, -15), textcoords='offset points', 
                ha='center', fontsize=9, fontweight='bold')

# Highlight optimal point (12 frames)
ax1.axvline(x=12, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax1.text(12.2, 81, 'Optimal\n12 frames', fontsize=11, color='red', fontweight='bold')

# Add improvement annotation
ax1.annotate(f'86% → 95% accuracy\n0.17s → 0.40s latency', 
            xy=(12, 95), xytext=(14, 90),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower right')

plt.title('Buffer Size Optimization: Accuracy vs Latency Trade-off', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('buffer_optimization.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================
# GRAPH 2: FALSE POSITIVE REDUCTION
# =============================================
plt.figure(figsize=(10, 6))

# Your false positive data
false_positives = [8.3, 5.2, 3.1, 1.2, 1.0, 0.9, 0.8, 0.7]

# Create bar chart
bars = plt.bar([str(bs) for bs in buffer_sizes], false_positives, 
               color=['#ff9999', '#ff9999', '#ff9999', '#66b3ff', 
                      '#66b3ff', '#66b3ff', '#66b3ff', '#66b3ff'],
               edgecolor='black', linewidth=1)

plt.xlabel('Buffer Size (frames)', fontsize=12, fontweight='bold')
plt.ylabel('False Positives per Minute', fontsize=12, fontweight='bold')
plt.title('False Positive Reduction with Buffer Size', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add value labels on bars
for bar, fp in zip(bars, false_positives):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
             f'{fp:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Add reduction arrow and text
reduction = ((false_positives[0] - false_positives[3]) / false_positives[0]) * 100
plt.annotate(f'{reduction:.0f}% Reduction!', 
            xy=(3.5, false_positives[3]), 
            xytext=(2, false_positives[0]),
            arrowprops=dict(arrowstyle='->', color='green', lw=2.5),
            fontsize=12, fontweight='bold', color='green',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

# Highlight optimal point
plt.axvline(x=3.5, color='red', linestyle='--', alpha=0.5)  # Between 10 and 12
plt.text(3.8, 7.5, 'Optimal\n12 frames', fontsize=11, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('false_positives.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================
# GRAPH 3: FPS STABILITY (Fixed with your actual 1 FPS data)
# =============================================
# Since your system shows 1 FPS, we'll create realistic data
# but you should replace this with your actual logged data

# Simulated 1 FPS data (replace with your actual logs)
time_seconds = np.arange(0, 60, 1)  # 60 seconds
fps_values = np.ones_like(time_seconds)  # Constant 1 FPS
# Add tiny variation for realism
fps_values = fps_values + np.random.normal(0, 0.05, len(time_seconds))
fps_values = np.clip(fps_values, 0.95, 1.05)

plt.figure(figsize=(12, 5))

# Subplot 1: FPS over time
plt.subplot(1, 2, 1)
plt.plot(time_seconds, fps_values, 'g-', linewidth=1.5)
plt.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='Target 30 FPS')
plt.axhline(y=np.mean(fps_values), color='b', linestyle='--', alpha=0.5, 
            label=f'Avg {np.mean(fps_values):.1f} FPS')
plt.xlabel('Time (seconds)', fontsize=11)
plt.ylabel('FPS', fontsize=11)
plt.title('FPS Stability Over Time', fontsize=12, fontweight='bold')
plt.ylim(0, 35)
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 2: FPS Distribution
plt.subplot(1, 2, 2)
plt.hist(fps_values, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(x=30, color='r', linestyle='--', alpha=0.5, label='Target 30 FPS')
plt.axvline(x=np.mean(fps_values), color='b', linestyle='--', alpha=0.5, 
            label=f'Avg {np.mean(fps_values):.1f} FPS')
plt.xlabel('FPS', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('FPS Distribution', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

plt.suptitle('Real-Time Processing Performance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fps_stability.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================
# PRINT SUMMARY TABLES
# =============================================
print("\n" + "="*70)
print("📊 BUFFER OPTIMIZATION SUMMARY")
print("="*70)
print(f"{'Buffer':<12} {'Accuracy':<15} {'Latency':<15} {'False Positives':<20}")
print("-" * 62)
for bs, acc, lat, fp in zip(buffer_sizes, accuracy, latency, false_positives):
    print(f"{bs:<12} {acc:.1f}%{' ':<11} {lat:.2f}s{' ':<11} {fp:.1f}/min")
print("-" * 62)
print(f"\n✅ Optimal buffer size: 12 frames")
print(f"✅ Accuracy improvement: {accuracy[3]-accuracy[0]:.1f}% (86% → 95%)")
print(f"✅ False positive reduction: {reduction:.0f}% (8.3 → 1.2/min)")
print(f"✅ Latency at optimal: 0.40s")
print("="*70)

print("\n📈 FPS Performance Summary:")
print(f"  Current FPS: 1.0 FPS")
print(f"  Target FPS: 30 FPS")
print(f"  Note: FPS is low - consider optimizing processing pipeline")