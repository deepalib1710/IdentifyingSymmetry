import numpy as np
import matplotlib.pyplot as plt

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []

    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []

        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)

        path_XYs.append(XYs)

    return path_XYs

def plot(path_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colors = ['black']  

    for i, XYs in enumerate(path_XYs):
        c = colors[i % len(colors)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=4, marker='o', markersize=10)
        intersection_x = XYs[0][1, 0]
        intersection_y = XYs[0][1, 1]
        ax.scatter(intersection_x, intersection_y, color='black', marker='o', s=70)

    ax.set_aspect('equal')
    plt.show()

csv_path = "C:/Users/Krishna Kumar Banka/Downloads/problems/problems/frag2.csv"
result = read_csv(csv_path)
plot(result)
