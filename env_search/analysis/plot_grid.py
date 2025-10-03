import fire
import matplotlib.pyplot as plt


from env_search.utils import (read_in_sortation_map, sortation_env_number2str,
                              sortation_env_str2number, get_chute_loc,
                              get_workstation_loc, sortation_obj_types, DIRS)

def plot_grid(map_filepath: str):
    map_str, map_name = read_in_sortation_map(map_filepath)
    map_np = sortation_env_str2number(map_str)
    h, w = map_np.shape
    chute_locs = get_chute_loc(map_np, flatten=False)

    # # --- Parameters you can change ---
    # grid_size = 5  # Creates a 5x5 grid
    # cell_values = {
    #     (0, 0): 10,   # (row, col): number
    #     (1, 3): 5,
    #     (3, 2): 7,
    #     (4, 4): 12
    # }

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(20, 12))

    for row in range(h + 1):
        ax.axhline(row, color='black', linewidth=1)
    for col in range(w + 1):
        ax.axvline(col, color='black', linewidth=1)

    # Place the numbers in specified cells
    # Note: by default, row=0 is at the bottom. 
    # If you want row=0 at the top, you can invert the y-axis (see below).
    for (row, col) in chute_locs:
        # breakpoint()
        idx = (h - row - 1) * w + col
        ax.text(
            col + 0.5,    # X position (center of cell horizontally)
            row + 0.5,    # Y position (center of cell vertically)
            str(idx),  # Text to display
            ha='center',  # Horizontal alignment
            va='center',  # Vertical alignment
            fontsize=10   # Font size
        )

    # Set the plot limits so that the grid fits exactly
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)

    # Optionally flip the y-axis if you want row=0 at the top
    # ax.invert_yaxis()

    # Remove the axis tick marks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("grid.png")

if __name__ == "__main__":
    fire.Fire(plot_grid)