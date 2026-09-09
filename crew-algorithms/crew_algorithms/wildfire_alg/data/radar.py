import matplotlib.pyplot as plt
import numpy as np

GRID_ALPHA = 0.5

def make_radar_chart(df, title):
    categories = df['Behaviour'].tolist()
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    cmap = plt.get_cmap('tab10')  # a palette of 10 distinct colors

    for idx, col in enumerate(df.columns[1:]):
        values = df[col].tolist()
        values += values[:1]

        ax.plot(
            angles,
            values,
            label=col,
            color=cmap(idx),
            linewidth=2.5,
            linestyle=['dotted','dotted','dotted','dotted','dotted'][idx],
            marker='o',
            markersize=6,
        )
        ax.fill(angles, values, color=cmap(idx), alpha=0.1)

    

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title, y=1.08)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=GRID_ALPHA)
    ax.spines['polar'].set_color('gray')
    ax.spines['polar'].set_linestyle('--')
    ax.spines['polar'].set_linewidth(0.5)
    ax.spines['polar'].set_alpha(GRID_ALPHA)
    ax.set_xticklabels(categories, fontsize=12)
    for label in ax.get_yticklabels():
        label.set_fontsize(20)


    # place legend to the right of the plot
    ax.legend(
        bbox_to_anchor=(1.3, 1.0),
        loc='upper left',
        borderaxespad=0.,
        fontsize=20
    )

    # tighten layout so nothing is cut off
    plt.tight_layout()
    # after you’ve plotted everything, but before plt.show():
    # collect all (non‐zero) values:
    all_vals = df.iloc[:,1:].values.flatten()
    min_val = 0
    max_val = 1

    # pad a little bit on each end:
    lower = 0 # 10% below your min
    upper = 1   # 10% above your max

    ax.set_ylim(lower, upper)
    ax.set_yticks(np.linspace(lower, upper, 5))
    ax.set_yticklabels([f"{v:.2f}" for v in np.linspace(lower, upper, 5)])

        
    plt.show()




def save_highdef_radar(df, title, filename, dpi=300, figsize=(8, 8)):
    # Prepare angles
    categories = df['Behaviour'].tolist()
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Create figure with desired size
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    cmap = plt.get_cmap('tab10')

    # Plot each series
    for idx, col in enumerate(df.columns[1:]):
        values = df[col].tolist()
        values += values[:1]
        ax.plot(
            angles,
            values,
            label=col,
            color=cmap(idx),
            linewidth=2.5,
            linestyle=['dotted','dotted','dotted','dotted','dotted'][idx],
            marker='o',
            markersize=6,
        )
        ax.fill(angles, values, color=cmap(idx), alpha=0.1)

    # Ticks and title
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    for label in ax.get_xticklabels():
        label.set_fontsize(16)
    ax.set_title(title, y=1.08)
    for label in ax.get_yticklabels():
        label.set_fontsize(16)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=GRID_ALPHA)
    ax.spines['polar'].set_color('gray')
    ax.spines['polar'].set_linestyle('--')
    ax.spines['polar'].set_linewidth(0.5)
    ax.spines['polar'].set_alpha(GRID_ALPHA)

    # Legend outside
    ax.legend(
        bbox_to_anchor=(1, 1.0),
        loc='upper left',
        borderaxespad=0.,
        fontsize=18
    )
    ax.set_ylim(0, 1)

    # Save as high-def PNG
    plt.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
