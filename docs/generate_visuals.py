"""
generate_blog_visuals.py — Generate all visualizations for the blog post.

Creates:
1. Peak torque comparison
2. Language-energy mapping
3. Success vs energy table visualization
4. Training metrics if available
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Outputs go to docs/images/
OUTPUT_DIR = "docs/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color scheme matching your existing pareto plot
DESCRIPTOR_COLORS = {
    "carefully":   "#5e81f4",
    "gently":      "#48c9b0",
    "efficiently": "#f0b429",
    "normally":    "#e07b39",
    "quickly":     "#e84393",
}

DESCRIPTOR_ORDER = ["carefully", "gently", "efficiently", "normally", "quickly"]


def create_peak_torque_comparison():
    """Bar chart: Peak torque by language command."""
    door_df = pd.read_csv('results/door_pareto_table.csv')
    
    # Sort by descriptor order
    door_df['descriptor_ord'] = door_df['descriptor'].map(
        {d: i for i, d in enumerate(DESCRIPTOR_ORDER)}
    )
    door_df = door_df.sort_values('descriptor_ord')
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = [DESCRIPTOR_COLORS.get(d, '#999999') for d in door_df['descriptor']]
    bars = ax.bar(door_df['descriptor'], door_df['peak_torque'], 
                   color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax.set_ylabel('Peak Torque (N·m)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Language Command', fontsize=12, fontweight='bold')
    ax.set_title('Peak Torque Varies by Language Command', fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/peak_torque_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Created: {OUTPUT_DIR}/peak_torque_comparison.png")
    plt.close()


def create_language_energy_mapping():
    """Horizontal bar chart: Energy budget by language command."""
    door_df = pd.read_csv('results/door_pareto_table.csv')
    
    # Sort by energy (ascending)
    door_df = door_df.sort_values('total_energy')
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Color by success rate
    colors = []
    for _, row in door_df.iterrows():
        if row['success'] > 0.8:
            colors.append('#2ecc71')  # Green for high success
        elif row['success'] > 0.5:
            colors.append('#f39c12')  # Orange for medium
        else:
            colors.append('#e74c3c')  # Red for low success
    
    bars = ax.barh(door_df['descriptor'], door_df['total_energy'], 
                    color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax.set_xlabel('Energy per Episode (Joules)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Language Command', fontsize=12, fontweight='bold')
    ax.set_title('Energy Budget Controlled by Language', fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add value labels
    for bar, (_, row) in zip(bars, door_df.iterrows()):
        width = bar.get_width()
        success_pct = row['success'] * 100
        ax.text(width + 10, bar.get_y() + bar.get_height()/2.,
                f'{width:.0f}J ({success_pct:.0f}% success)',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='High success (>80%)'),
        Patch(facecolor='#f39c12', edgecolor='black', label='Medium success (50-80%)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Low success (<50%)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/language_energy_mapping.png', dpi=300, bbox_inches='tight')
    print(f"✓ Created: {OUTPUT_DIR}/language_energy_mapping.png")
    plt.close()


def create_metrics_comparison_table():
    """Create a visual comparison table as an image."""
    door_df = pd.read_csv('results/door_pareto_table.csv')
    
    # Sort by descriptor order
    door_df['descriptor_ord'] = door_df['descriptor'].map(
        {d: i for i, d in enumerate(DESCRIPTOR_ORDER)}
    )
    door_df = door_df.sort_values('descriptor_ord')
    
    # Calculate energy savings relative to "normally"
    baseline_energy = door_df[door_df['descriptor'] == 'normally']['total_energy'].values[0]
    door_df['energy_savings'] = ((baseline_energy - door_df['total_energy']) / baseline_energy * 100)
    
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Command', 'Success Rate', 'Energy (J)', 'vs Baseline', 'Peak Torque (N·m)']
    
    for _, row in door_df.iterrows():
        success_str = f"{row['success']*100:.0f}%"
        if row['success'] >= 0.8:
            success_str += " ✓"
        elif row['success'] < 0.5:
            success_str += " ×"
        
        savings_str = f"{row['energy_savings']:+.0f}%"
        if row['descriptor'] == 'normally':
            savings_str = "baseline"
        
        table_data.append([
            row['descriptor'].capitalize(),
            success_str,
            f"{row['total_energy']:.0f}",
            savings_str,
            f"{row['peak_torque']:.0f}"
        ])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    # Style rows - alternate colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#ecf0f1')
            else:
                cell.set_facecolor('#ffffff')
            
            # Highlight "normally" row
            if table_data[i-1][0] == 'Normally':
                cell.set_facecolor('#fff3cd')
                cell.set_text_props(weight='bold')
    
    plt.title('Door Opening Task: Performance by Language Command', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(f'{OUTPUT_DIR}/metrics_comparison_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ Created: {OUTPUT_DIR}/metrics_comparison_table.png")
    plt.close()


def create_success_energy_scatter():
    """Scatter plot showing the tradeoff more clearly."""
    door_df = pd.read_csv('results/door_pareto_table.csv')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for _, row in door_df.iterrows():
        color = DESCRIPTOR_COLORS.get(row['descriptor'], '#999999')
        ax.scatter(row['total_energy'], row['success'] * 100, 
                   s=300, color=color, edgecolor='black', linewidth=2,
                   alpha=0.85, zorder=5)
        
        # Add label
        ax.annotate(row['descriptor'].capitalize(),
                    (row['total_energy'], row['success'] * 100),
                    xytext=(10, 0), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3, edgecolor='none'))
    
    ax.set_xlabel('Energy per Episode (Joules)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('The Success-Efficiency Tradeoff', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(door_df['total_energy']) * 1.15)
    ax.set_ylim(-5, 105)
    
    # Add annotation about tradeoff
    ax.text(0.95, 0.05, 
            'Lower energy = more efficient\nHigher success = more reliable\nYou can\'t optimize both simultaneously',
            transform=ax.transAxes, fontsize=9, style='italic',
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/success_energy_tradeoff.png', dpi=300, bbox_inches='tight')
    print(f"✓ Created: {OUTPUT_DIR}/success_energy_tradeoff.png")
    plt.close()


if __name__ == "__main__":
    print("\n🎨 Generating blog post visuals...\n")
    
    create_peak_torque_comparison()
    create_language_energy_mapping()
    create_metrics_comparison_table()
    create_success_energy_scatter()
    
    print("\n✅ All visuals generated successfully!")
    print(f"📁 Saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - peak_torque_comparison.png")
    print("  - language_energy_mapping.png")
    print("  - metrics_comparison_table.png")
    print("  - success_energy_tradeoff.png")
