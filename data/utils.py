import numpy as np
import matplotlib.pyplot as plt


def dec_conflict(g0, g1):
    for ele in g0:
        if ele in g1:
            return False
    return True


def plot_overlap_conflict(votes, fid2clusters):
    plot_overlap(votes)
    plot_conflict(votes, fid2clusters)


def plot_overlap(votes):
    num_plfs = votes.shape[1]
    overlaps = np.zeros((num_plfs, num_plfs))
    for i in range(num_plfs):
        for j in range(i, num_plfs):
            total = 0
            for example in range(len(votes)):
                vote_i = votes[example, i]
                vote_j = votes[example, j]
                if vote_i != -1 and vote_j != -1:
                    overlaps[i, j] += 1
    plot_heatmap(overlaps, None, title="Overlaps",
                 colorbar_title="Num. overlapping votes")


def plot_conflict(votes, fid2clusters):
    num_plfs = votes.shape[1]
    conflicts = np.zeros((num_plfs, num_plfs))
    for i in range(num_plfs):
        for j in range(i, num_plfs):
            total = 0
            for example in range(len(votes)):  # instance
                vote_i = votes[example, i]
                vote_j = votes[example, j]
                if vote_i != -1 and vote_j != -1:
                    if dec_conflict(fid2clusters[i][int(vote_i)], fid2clusters[j][int(vote_j)]):
                        conflicts[i, j] += 1
    plot_heatmap(conflicts, None, title="Conflicts",
                 colorbar_title="Num. conflicting votes")


def plot_heatmap(heatmap, ax_names=None, title="", colorbar_title=""):
    """
    Based on code from
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """
    fig, ax = plt.subplots()
    im = ax.imshow(heatmap)
    # We want to show all ticks...
    ax.set_xticks(np.arange(heatmap.shape[0]))
    ax.set_yticks(np.arange(heatmap.shape[1]))
    # ... and label them with the respective list entries
    if ax_names is not None:
        ax.set_xticklabels(ax_names)
        ax.set_yticklabels(ax_names)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(colorbar_title, rotation=-90, va="bottom")
    # Create title
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def votes_filter(votes, labels=None, return_idx=False, ratio=1):
    filtered_votes = []
    filtered_labels = []
    filtered_idx = []
    votes = np.array(votes)
    num_plfs = votes.shape[1]

    for idx, rows in enumerate(votes):
        if len(np.where(rows == -1)[0]) / num_plfs < ratio:
            filtered_votes.append(rows)
            if labels is not None:
                filtered_labels.append(labels[idx])
            filtered_idx.append(idx)
    if return_idx:
        return np.array(filtered_votes), filtered_labels, filtered_idx
    return np.array(filtered_votes), filtered_labels