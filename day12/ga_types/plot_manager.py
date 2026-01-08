""" Manages plotting of graphs"""
import matplotlib.pyplot as plt


class InteractiveIntPlotter:
    """ Class to visualise evolution of genetic algorithm"""

    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        # White(0) → Green(1) → Red(>1)
        colors = [(1, 1, 1), (0, 0.8, 0), (1, 0, 0)]
        self.cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
            'grad', colors, 256)
        self.img = None

    def update(self, data):
        """Update plot with new 2D integer array"""
        if self.img is None:
            self.img = self.ax.imshow(
                data, cmap=self.cmap, vmin=0, vmax=max(2, data.max()))
            plt.colorbar(self.img, ax=self.ax)
        else:
            self.img.set_data(data)
            self.img.set_clim(vmax=max(2, data.max()))

        self.ax.set_title(f"Max value: {data.max()}")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
