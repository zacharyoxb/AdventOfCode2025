""" Manages plotting of graphs"""
from numpy.typing import NDArray
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


class Plotter:
    """ Class to visualise evolution of genetic algorithm """

    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.ion()

        # White(0) -> Green(1) -> Gradient to Red (>1)
        self.colors = ["#FFFFFF", "#008000", "#FF0000"]
        self.img = None
        self.cbar = None

        # Set up the figure
        self.ax.set_title("GA Evolution")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)  # Give time for window to appear

    def _get_colormap(self, max_val: int) -> plt.cm.colors.LinearSegmentedColormap:
        """Create colormap based on max value"""
        # Ensure max_val is at least 2
        max_val = max(2, max_val)

        # Normalize the values to [0, 1]
        # 0 -> white, 0.5 -> green, 1 -> red (for max_val >= 2)
        if max_val == 1:
            # Simple white to green gradient
            return plt.cm.colors.LinearSegmentedColormap.from_list(
                "custom", [(0, "#FFFFFF"), (1, "#008000")])
        # White(0) -> Green(mid) -> Red(1)
        # Map: 0 -> white, 1/max_val -> green, 1 -> red
        green_pos = 1.0 / max_val
        return plt.cm.colors.LinearSegmentedColormap.from_list(
            "custom",
            [(0, "#FFFFFF"), (green_pos, "#008000"), (1, "#FF0000")]
        )

    def update(self, data: NDArray[np.int32]):
        """ Update plot with new 2D integer array """
        max_val = int(np.max(data))
        cmap = self._get_colormap(max_val)

        if self.img is None:
            self.img = self.ax.imshow(
                data,
                cmap=cmap,
                vmin=0,
                vmax=max(1, max_val),
                interpolation='nearest'
            )
            # Add colorbar
            self.cbar = self.fig.colorbar(self.img, ax=self.ax)
            self.cbar.set_label('Value')
        else:
            # Update existing image
            self.img.set_data(data)
            self.img.set_cmap(cmap)
            self.img.set_clim(vmin=0, vmax=max(1, max_val))

        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # Small pause to allow UI updates

    def close(self):
        """ Close the plotter properly """
        if hasattr(self, 'fig') and self.fig:
            plt.close(self.fig)

    def __del__(self):
        """ Destructor """
        self.close()
