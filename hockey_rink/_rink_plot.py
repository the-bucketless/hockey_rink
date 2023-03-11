""" Module for adding plotting functions to BaseRink.

Not intended for direct use, only as a parent class.
"""


from functools import wraps
from hockey_rink._base_rink import BaseRink
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np
import re
from scipy.stats import binned_statistic_2d


class BaseRinkPlot(BaseRink):
    """ Class extending BaseRink to include plotting methods. """

    def _validate_values(plot_function):
        """ Decorator to restrict x, y, and values to intended coordinates.

        Expects x and y to have already been converted for plotting.
        """

        @wraps(plot_function)
        def wrapper(self, x, y, *, values=None,
                    plot_range=None, plot_xlim=None, plot_ylim=None, **kwargs):

            C = kwargs.pop("C", None)
            values = C if values is None else values

            values = self.copy_(values)

            # x and y will already have been symmetrized
            if values is None:
                values = np.ones_like(x)
            else:
                values = np.ravel(values)

                if kwargs.get("symmetrize", False):
                    values = np.concatenate((values, values))

            if len(x) != len(y) or len(x) != len(values):
                raise Exception("x, y, and values must have equal length.")

            mask = False

            if plot_range is None and plot_xlim is None and plot_ylim is None:
                plot_xlim, plot_ylim = self._get_limits("full")
            else:
                plot_xlim, plot_ylim = self._get_limits(plot_range, plot_xlim, plot_ylim)

                mask = ((x < plot_xlim[0]) | (x > plot_xlim[1])
                        | (y < plot_ylim[0]) | (y > plot_ylim[1]))

            is_constrained = kwargs.get("is_constrained", True)
            if is_constrained:
                values = self._outside_rink_to_nan(x, y, values)

            mask = mask | np.isnan(x) | np.isnan(y) | np.isnan(values)

            x = x[~mask]
            y = y[~mask]
            values = values[~mask]

            if not is_constrained:
                plot_xlim = [min([*plot_xlim, *x]), max([*plot_xlim, *x])]
                plot_ylim = [min([*plot_ylim, *y]), max([*plot_ylim, *y])]

            return plot_function(self, x, y, values=values,
                                 plot_range=plot_range,
                                 plot_xlim=plot_xlim, plot_ylim=plot_ylim,
                                 **kwargs)

        return wrapper

    def _validate_plot(plot_function):
        """ Decorator to ensure all plotting parameters are of the correct form. """

        @wraps(plot_function)
        def wrapper(self, *args, **kwargs):
            if "ax" not in kwargs:
                kwargs["ax"] = plt.gca()

            args = list(args)
            for coord in ("x", "y", "x1", "y1", "x2", "y2"):
                if coord in kwargs:
                    args.append(kwargs.pop(coord))

            for i in range(len(args)):
                args[i] = self.copy_(args[i])
                args[i] = np.ravel(args[i])

                is_y = i % 2

                if kwargs.get("symmetrize", False):
                    args[i] = np.concatenate((args[i], args[i] * (-1 if is_y else 1)))

                args[i] = args[i] - (self.y_shift if is_y else self.x_shift)

            kwargs["transform"] = self._get_transform(kwargs["ax"])

            # avoid cutting off markers when plotting outside of rink
            kwargs["clip_on"] = kwargs.get("clip_on", kwargs.get("is_constrained", True))

            args = tuple(args)

            return plot_function(self, *args, **kwargs)

        return wrapper

    def _constrain_plot(self, plot_features, ax, transform):
        """ Constrain the features of a plot to only display inside the boards. """

        try:
            iter(plot_features)
        except TypeError:
            plot_features = [plot_features]

        boards_x, boards_y = self._boards.get_xy_for_clip()
        constraint = plt.Polygon(tuple(zip(boards_x, boards_y)), transform=transform)

        for plot_feature in plot_features:
            plot_feature.set_clip_path(constraint)

    def _outside_rink_to_nan(self, x, y, values):
        """ Set values of coordinates outside the boundaries of the rink to nan. """

        x = np.abs(x).astype("float32")
        y = np.abs(y).astype("float32")

        values = values.astype("float32")

        center_x = self._boards.length / 2 - self._boards.radius
        center_y = self._boards.width / 2 - self._boards.radius

        mask = ((x > center_x) & (y > center_y)
                & ((center_x - x) ** 2 + (center_y - y) ** 2 > self._boards.radius ** 2))
        values[mask] = np.nan

        return values

    def _bound_rink(self, x, y, plot_features, ax, transform, is_constrained, update_display_range):
        """ Update board contraints and limits of plot. """

        if is_constrained:
            self._constrain_plot(plot_features, ax, transform)
        else:
            if update_display_range:
                self._update_display_range(x, y, ax)

    @staticmethod
    def binned_stat_2d(x, y, values, statistic="sum", xlim=None, ylim=None, binsize=1, bins=None):
        """ Use scipy to compute a bi-dimensional binned statistic.

        Parameters:
            x: array_like

            y: array_like

            values: array_like

            statistic: string or callable; default: "sum"
                The statistic to compute via scipy.  The following are available:
                    "mean": the mean of values for points within each bin.
                    "std": the standard deviation within each bin.
                    "median": the median of values for points within each bin.
                    "count": the number of points within each bin.
                    "sum": the sum of values for points within each bin.
                    "min": the minimum of values for points within each bin.
                    "max": the maximum of values for points within each bin.
                    function: a user-defined function which takes a 1D array of values and outputs a
                        single numerical statistic.

            xlim: (float, float); optional
                The lower and upper bounds of the x-coordinates to include.

                Only used if bins is None.

            ylim: (float, float); optional
                The lower and upper bounds of the y-coordinates to include.

                Only used if bins is None.

            binsize: float or (float, float); default: 1
                The size of the bins for a given portion of the rink.
                    float: the size of the bins for the two dimensions.
                    (float, float): the size of the bins in each dimension.

                Only used if bins is None.

            bins: int or (int, int) or array_like or (array, array); optional
                The bin specification:
                    int: the number of bins for the two dimensions.
                    (int, int): the number of bins in each dimension.
                    array_like: the bin edges for the two dimensions.
                    (array, array): the bin edges in each dimension.

        Returns:
            stat: (nx, ny) ndarray
                The values of the selected statistic in each two-dimensional bin.
            x_edge: (nx + 1) ndarray
                The bin edges along the first dimension.
            y_edge: (ny + 1) ndarray
                The bin edges along the second dimension.
        """

        if bins is None:
            try:
                iter(binsize)
            except TypeError:
                binsize = (binsize, binsize)

            # coordinates set as center of each bin
            x_edge = np.arange(xlim[0] - binsize[0] / 2, xlim[1] + binsize[0], binsize[0])
            y_edge = np.arange(ylim[0] - binsize[1] / 2, ylim[1] + binsize[1], binsize[1])

            bins = [x_edge, y_edge]

        stat, x_edge, y_edge, _ = binned_statistic_2d(x, y, values, statistic=statistic, bins=bins)

        stat = stat.T

        return stat, x_edge, y_edge

    def constrain_plot(self, ax=None, collection=None):
        """ Constrain a collection object to only display inside the boards.

        Parameters:
            ax: matplotlib Axes; optional
                Axes in which to draw the plot.  If not provided, will use the currently active Axes.

            collection: matplotlib collection or iterable of matplotlib collections; default: None
                The collection to be constrained.

                If None, will constrain all collections found on the Axes.
        """

        ax = plt.gca() if ax is None else ax
        transform = self._get_transform(ax)

        if collection is None:
            collection = ax.collections

        self._constrain_plot(collection, ax, transform)

    @_validate_plot
    def plot(self, x, y, *, is_constrained=True, update_display_range=False, zorder=20, ax=None, **kwargs):
        """ Wrapper for matplotlib plot function.

        Will plot to areas out of view when full ice surface is not displayed.

        All parameters other than x and y require keywords.
            ie) plot(x, y, False) won't work, needs to be plot(x, y, is_constrained=False)

        Parameters:
            x: array_like

            y: array_like

            is_constrained: bool; default: True
                Indicates whether or not the plot is constrained to remain inside the boards.

            update_display_range: bool; default: False
                Indicates whether or not to update the display range when coordinates are outside
                the given range. Only used when is_constrained is False.

            zorder: float; default: 20
                Determines which rink features the plot will draw over.

            ax: matplotlib Axes; optional
                Axes in which to draw the plot.  If not provided, will use the currently active Axes.

            **kwargs: Any other matplotlib plot properties; optional

        Returns:
            list of matplotlib Line2D
        """

        img = ax.plot(x, y, zorder=zorder, **kwargs)
        self._bound_rink(x, y, img, ax, kwargs["transform"], is_constrained, update_display_range)
        return img

    @_validate_plot
    def scatter(self, x, y, *, is_constrained=True, update_display_range=False, symmetrize=False,
                zorder=20, ax=None, **kwargs):
        """ Wrapper for matplotlib scatter function.

        Will plot to areas out of view when full ice surface is not displayed.

        All parameters other than x and y require keywords.
            ie) scatter(x, y, False) won't work, needs to be scatter(x, y, is_constrained=False)

        Parameters:
            x: array_like

            y: array_like

            is_constrained: bool; default: True
                Indicates whether or not the plot is constrained to remain inside the boards.

            update_display_range: bool; default: False
                Indicates whether or not to update the display range when coordinates are outside
                the given range. Only used when is_constrained is False.

            symmetrize: bool; default: False
                Indicates whether or not to reflect the coordinates across the y-axis.

            zorder: float; default: 20
                Determines which rink features the plot will draw over.

            ax: matplotlib Axes; optional
                Axes in which to draw the plot.  If not provided, will use the currently active Axes.

            **kwargs: Any other matplotlib scatter properties; optional

        Returns:
            matplotlib PathCollection
        """

        img = ax.scatter(x, y, zorder=zorder, **kwargs)
        self._bound_rink(x, y, img, ax, kwargs["transform"], is_constrained, update_display_range)
        return img

    @_validate_plot
    def arrow(self, x1, y1, x2, y2, *, is_constrained=True, update_display_range=False,
              length_includes_head=True, head_width=1,
              zorder=20, ax=None, **kwargs):
        """ Wrapper for matplotlib arrow function.

        Will plot to areas out of view when full ice surface is not displayed.

        All parameters other than x1, y1, x2, and y2 require keywords.
            ie) arrow(x1, y1, x2, y2, False) won't work, needs to be arrow(x1, y1, x2, y2, is_constrained=False)

        Parameters:
            x1: array_like
                The starting x-coordinates of the arrows.

            y1: array_like
                The starting y-coordinates of the arrows.

            x2: array_like
                The ending x-coordinates of the arrows.

            y2: array_like
                The ending y-coordinates of the arrows.

            is_constrained: bool; default: True
                Indicates whether or not the plot is constrained to remain inside the boards.

            update_display_range: bool; default: False
                Indicates whether or not to update the display range when coordinates are outside
                the given range. Only used when is_constrained is False.

            length_includes_head: bool; default: True
                Indicates if head of the arrow is to be included in calculating the length.

            head_width: float or None; default: 1
                Total width of the full arrow head.

            zorder: float; default: 20
                Determines which rink features the plot will draw over.

            ax: matplotlib Axes; optional
                Axes in which to draw the plot.  If not provided, will use the currently active Axes.

            **kwargs: Any other matplotlib arrow properties; optional

        Returns:
            list of matplotlib FancyArrow
        """

        dx = x2 - x1
        dy = y2 - y1

        arrows = []
        for i in range(len(x1)):
            arrows.append(ax.arrow(x1[i], y1[i], dx[i], dy[i],
                                   zorder=zorder, head_width=head_width,
                                   length_includes_head=length_includes_head, **kwargs))

        self._bound_rink(
            [*x1, *x2], [*y1, *y2], arrows, ax,
            kwargs["transform"], is_constrained, update_display_range
        )

        return arrows

    @_validate_plot
    @_validate_values
    def hexbin(self, x, y, *, values=None, is_constrained=True, update_display_range=False, symmetrize=False,
               plot_range=None, plot_xlim=None, plot_ylim=None,
               gridsize=None, binsize=1, zorder=2, clip_on=True, ax=None, **kwargs):
        """ Wrapper for matplotlib hexbin function.

        Will plot to areas out of view when full ice surface is not displayed.
        Use plot_range, plot_xlim, and plot_ylim to restrict to the area within view.

        All parameters other than x and y require keywords.
            ie) hexbin(x, y, values) won't work, needs to be hexbin(x, y, values=values)

        Parameters:
            x: array_like

            y: array_like

            values: array_like; optional
                If None, values of 1 will be assigned to each x,y-coordinate provided.

            is_constrained: bool; default: True
                Indicates whether or not the plot is constrained to remain inside the boards.

                If plot ranges are used, also constrains coordinates included to remain inside the boards.

            update_display_range: bool; default: False
                Indicates whether or not to update the display range when coordinates are outside
                the given range. Only used when is_constrained is False.

            symmetrize: bool; default: False
                Indicates whether or not to reflect the coordinates and values across the y-axis.

            plot_range: {"full", "half", "offense", "defense", "ozone", "dzone"}; optional
                Restricts the portion of the rink that can be plotted to.  Does so by removing values outside of
                the given range.

                Only affects x-coordinates and can be used in conjunction with ylim, but will be superceded by
                xlim if provided.

                "full": The entire length of the rink is displayed.
                "half" or "offense": The offensive half (largest x-coordinates) of the rink is displayed.
                "defense": The defensive half (smallest x-coordinates) of the rink is displayed.
                "ozone": The offensive zone (blue line to end boards) of the rink is displayed.
                "dzone": The defensive zone (end boards to blue line) of the rink is displayed.

                If no values are given for plot_range, plot_xlim, or plot_ylim, will use all coordinates provided.

            plot_xlim: float or (float, float); optional
                The range of x-coordinates to include in the plot.
                    float: the lower bound of the x-coordinates.  The upper bound will be the boards.
                    (float, float): The lower and upper bounds of the x-coordinates.

                If no values are given for plot_range, plot_xlim, or plot_ylim, will use all coordinates provided.

            plot_ylim: float or (float, float); optional
                The range of y-coordinates to include in the plot.
                    float: the lower bound of the y-coordinates.  The upper bound will be the boards.
                    (float, float): The lower and upper bounds of the y-coordinates.

                If no values are given for plot_range, plot_xlim, or plot_ylim, will use all coordinates provided.

            gridsize: int or (int, int); optional
                The grid specification:
                    int: the number of hexagons in the x-direction.  The number of hexagons in the y-direction is
                        chosen such that hexagons are approximately regular.
                    (int, int): the number of hexagons in both directions.

            binsize: float or (float, float); default: 1
                The size of the bins for a given portion of the rink.
                    float: the size of the bins for the two dimensions.
                    (float, float): the size of the bins in each dimension.

            zorder: float; default: 2
                Determines which rink features the plot will draw over.

            clip_on: bool; default: True
                Whether the artist uses clipping.

                Other plotting features will automatically be set to the same value as is_constrained, but doing
                so can lead to odd results with hexbin.

            ax: matplotlib Axes; optional
                Axes in which to draw the plot.  If not provided, will use the currently active Axes.

            **kwargs: Any other matplotlib hexbin properties; optional

        Returns:
            matplotlib PolyCollection
        """

        # setting clip_on to False can lead to odd results
        kwargs["clip_on"] = kwargs.get("clip_on", True)

        try:
            iter(binsize)
        except:
            binsize = (binsize, binsize)

        default_gridsize = (int((plot_xlim[1] - plot_xlim[0]) / binsize[0]),
                            int((plot_ylim[1] - plot_ylim[0]) / binsize[1]))
        gridsize = gridsize or default_gridsize

        # matplotlib hexbin uses count when C is None, but uses np.mean when values are included
        # since None is replaced in values with an array of ones, the reduce function needs to be
        # changed to allow for the same default behaviour
        if np.all(values == 1):
            kwargs["reduce_C_function"] = kwargs.get("reduce_C_function", np.sum)

        # delay application of transform until after drawing hexbin
        # an update to matplotlib doesn't rotate the position of the hexagons
        transform = kwargs.pop("transform")
        hexagon_transform = transform - ax.transData
        img = ax.hexbin(x, y, C=values, gridsize=gridsize, zorder=zorder, **kwargs)
        hexagon = img.get_paths()[0]
        hexagon.vertices = hexagon_transform.transform(hexagon.vertices)
        img.set_offsets(hexagon_transform.transform(img.get_offsets()))

        self._bound_rink(x, y, img, ax, transform, is_constrained, update_display_range)

        return img

    @_validate_plot
    @_validate_values
    def heatmap(self, x, y, *, values=None, is_constrained=True, update_display_range=False, symmetrize=False,
                plot_range=None, plot_xlim=None, plot_ylim=None,
                statistic="sum", binsize=1, bins=None,
                zorder=2, ax=None, **kwargs):
        """ Wrapper for matplotlib pcolormesh function.

        Will plot to areas out of view when full ice surface is not displayed.
        Use plot_range, plot_xlim, and plot_ylim to restrict to the area within view.

        All parameters other than x and y require keywords.
            ie) heatmap(x, y, values) won't work, needs to be heatmap(x, y, values=values)

        Parameters:
            x: array_like

            y: array_like

            values: array_like; optional
                If None, values of 1 will be assigned to each x,y-coordinate provided.

            is_constrained: bool; default: True
                Indicates whether or not the plot is constrained to remain inside the boards.

                If plot ranges are used, also constrains coordinates included to remain inside the boards.

            update_display_range: bool; default: False
                Indicates whether or not to update the display range when coordinates are outside
                the given range. Only used when is_constrained is False.

            symmetrize: bool; default: False
                Indicates whether or not to reflect the coordinates and values across the y-axis.

            plot_range: {"full", "half", "offense", "defense", "ozone", "dzone"}; optional
                Restricts the portion of the rink that can be plotted to.  Does so by removing values outside of
                the given range.

                Only affects x-coordinates and can be used in conjunction with ylim, but will be superceded by
                xlim if provided.

                "full": The entire length of the rink is displayed.
                "half" or "offense": The offensive half (largest x-coordinates) of the rink is displayed.
                "defense": The defensive half (smallest x-coordinates) of the rink is displayed.
                "ozone": The offensive zone (blue line to end boards) of the rink is displayed.
                "dzone": The defensive zone (end boards to blue line) of the rink is displayed.

                If no values are given for plot_range, plot_xlim, or plot_ylim, will use all coordinates provided.

            plot_xlim: float or (float, float); optional
                The range of x-coordinates to include in the plot.
                    float: the lower bound of the x-coordinates.  The upper bound will be the boards.
                    (float, float): The lower and upper bounds of the x-coordinates.

                If no values are given for plot_range, plot_xlim, or plot_ylim, will use all coordinates provided.

            plot_ylim: float or (float, float); optional
                The range of y-coordinates to include in the plot.
                    float: the lower bound of the y-coordinates.  The upper bound will be the boards.
                    (float, float): The lower and upper bounds of the y-coordinates.

                If no values are given for plot_range, plot_xlim, or plot_ylim, will use all coordinates provided.

            statistic: string or callable; default: "sum"
                The statistic to compute via scipy.  The following are available:
                    "mean": the mean of values for points within each bin.
                    "std": the standard deviation within each bin.
                    "median": the median of values for points within each bin.
                    "count": the number of points within each bin.
                    "sum": the sum of values for points within each bin.
                    "min": the minimum of values for points within each bin.
                    "max": the maximum of values for points within each bin.
                    function: a user-defined function which takes a 1D array of values and outputs a
                        single numerical statistic.

            binsize: float or (float, float); default: 1
                The size of the bins for a given portion of the rink.
                    float: the size of the bins for the two dimensions.
                    (float, float): the size of the bins in each dimension.

                Only used if bins is None.

            bins: int or (int, int) or array_like or (array, array); optional
                The bin specification:
                    int: the number of bins for the two dimensions.
                    (int, int): the number of bins in each dimension.
                    array_like: the bin edges for the two dimensions.
                    (array, array): the bin edges in each dimension.

            zorder: float; default: 2
                Determines which rink features the plot will draw over.

            ax: matplotlib Axes; optional
                Axes in which to draw the plot.  If not provided, will use the currently active Axes.

            **kwargs: Any other matplotlib pcolormesh properties; optional

        Returns:
            matplotlib QuadMesh
        """

        stat, x_edge, y_edge = self.binned_stat_2d(x, y, values, statistic, plot_xlim, plot_ylim,
                                                   binsize, bins)

        img = ax.pcolormesh(x_edge, y_edge, stat, zorder=zorder, **kwargs)
        self._bound_rink(x, y, img, ax, kwargs["transform"], is_constrained, update_display_range)

        return img

    @_validate_plot
    @_validate_values
    def contour(self, x, y, *, values=None, fill=True,
                is_constrained=True, update_display_range=False, symmetrize=False,
                plot_range=None, plot_xlim=None, plot_ylim=None,
                statistic="sum", binsize=1, bins=None,
                zorder=2, ax=None, **kwargs):
        """ Wrapper for matplotlib contour and contourf functions.

        Will plot to areas out of view when full ice surface is not displayed.
        Use plot_range, plot_xlim, and plot_ylim to restrict to the area within view.

        All parameters other than x and y require keywords.
            ie) contour(x, y, values) won't work, needs to be contour(x, y, values=values)

        Parameters:
            x: array_like

            y: array_like

            values: array_like; optional
                If None, values of 1 will be assigned to each x,y-coordinate provided.

            fill: bool; default: True
                Indicates whether or not to fill in the contours.

            is_constrained: bool; default: True
                Indicates whether or not the plot is constrained to remain inside the boards.

                If plot ranges are used, also constrains coordinates included to remain inside the boards.

            update_display_range: bool; default: False
                Indicates whether or not to update the display range when coordinates are outside
                the given range. Only used when is_constrained is False.

            symmetrize: bool; default: False
                Indicates whether or not to reflect the coordinates and values across the y-axis.

            plot_range: {"full", "half", "offense", "defense", "ozone", "dzone"}; optional
                Restricts the portion of the rink that can be plotted to.  Does so by removing values outside of
                the given range.

                Only affects x-coordinates and can be used in conjunction with ylim, but will be superceded by
                xlim if provided.

                "full": The entire length of the rink is displayed.
                "half" or "offense": The offensive half (largest x-coordinates) of the rink is displayed.
                "defense": The defensive half (smallest x-coordinates) of the rink is displayed.
                "ozone": The offensive zone (blue line to end boards) of the rink is displayed.
                "dzone": The defensive zone (end boards to blue line) of the rink is displayed.

                If no values are given for plot_range, plot_xlim, or plot_ylim, will use all coordinates provided.

            plot_xlim: float or (float, float); optional
                The range of x-coordinates to include in the plot.
                    float: the lower bound of the x-coordinates.  The upper bound will be the boards.
                    (float, float): The lower and upper bounds of the x-coordinates.

                If no values are given for plot_range, plot_xlim, or plot_ylim, will use all coordinates provided.

            plot_ylim: float or (float, float); optional
                The range of y-coordinates to include in the plot.
                    float: the lower bound of the y-coordinates.  The upper bound will be the boards.
                    (float, float): The lower and upper bounds of the y-coordinates.

                If no values are given for plot_range, plot_xlim, or plot_ylim, will use all coordinates provided.

            statistic: string or callable; default: "sum"
                The statistic to compute via scipy.  The following are available:
                    "mean": the mean of values for points within each bin.
                    "std": the standard deviation within each bin.
                    "median": the median of values for points within each bin.
                    "count": the number of points within each bin.
                    "sum": the sum of values for points within each bin.
                    "min": the minimum of values for points within each bin.
                    "max": the maximum of values for points within each bin.
                    function: a user-defined function which takes a 1D array of values and outputs a
                        single numerical statistic.

            binsize: float or (float, float); default: 1
                The size of the bins for a given portion of the rink.
                    float: the size of the bins for the two dimensions.
                    (float, float): the size of the bins in each dimension.

                Only used if bins is None.

            bins: int or (int, int) or array_like or (array, array); optional
                The bin specification:
                    int: the number of bins for the two dimensions.
                    (int, int): the number of bins in each dimension.
                    array_like: the bin edges for the two dimensions.
                    (array, array): the bin edges in each dimension.

            zorder: float; default: 2
                Determines which rink features the plot will draw over.

            ax: matplotlib Axes; optional
                Axes in which to draw the plot.  If not provided, will use the currently active Axes.

            **kwargs: Any other matplotlib contour properties; optional

        Returns:
            matplotlib QuadContourSet
        """

        stat, x_edge, y_edge = self.binned_stat_2d(x, y, values, statistic, plot_xlim, plot_ylim,
                                                   binsize, bins)

        x_centers = (x_edge[:-1] + x_edge[1:]) / 2
        y_centers = (y_edge[:-1] + y_edge[1:]) / 2

        if plot_xlim is not None:
            x_centers[-1] = max(x_centers[-1], plot_xlim[1])
        if plot_ylim is not None:
            y_centers[-1] = max(y_centers[-1], plot_ylim[1])

        # avoid warning for argument not used in function
        kwargs.pop("clip_on", None)

        contour_function = ax.contourf if fill else ax.contour
        img = contour_function(x_centers, y_centers, stat, zorder=zorder, **kwargs)

        self._bound_rink(x, y, img.collections, ax, kwargs["transform"], is_constrained, update_display_range)

        return img

    """ Alias for contour. """
    contourf = contour

    def clear(self, ax=None, keep=None):
        """ Remove all plotted items from the rink. Can only be applied after drawing the rink.

        Parameters:
            ax: matplotlib Axes (optional)
                Axes to remove items from.  If not provided, will use the currently active Axes.

            keep: set (optional)
                Items that don't need to be removed.
        """

        if ax is None:
            ax = plt.gca()

        # Early exit if rink hasn't been drawn.
        if ax not in self._drawn:
            return

        keep = self._drawn[ax].union(np.ravel(keep))

        for child in ax.get_children():
            if child not in keep:
                child.remove()

    def get_plot_transform(self, ax=None, transform=None, include_transData=True):
        """ Return the matplotlib Transform to apply to plotted elements. """

        if ax is None:
            ax = plt.gca()

        shift = Affine2D().translate(-self.x_shift, -self.y_shift)
        rotation = self._rotations.get(ax, Affine2D().rotate_deg(self.rotation))

        rink_transform = shift + rotation

        if transform is not None:
            rink_transform += transform

        if include_transData:
            rink_transform += ax.transData

        return rink_transform

    def get_clip_path(self, ax=None, plot_range=None, plot_xlim=None, plot_ylim=None):
        """ The Polygon representing the clip path based on the outline of the boards subset to the desired
        plotting range.

        Arguments:
            ax: matplotlib Axes

            plot_range: {"full", "half", "offense", "defense", "ozone", "dzone"} (optional)
                Restricts the portion of the rink that can be plotted to beyond just the boards.

                Only affects x-coordinates and can be used in conjunction with ylim, but will be superceded by
                xlim if provided.

                "full": The entire length of the rink.
                "half" or "offense": The offensive half (largest x-coordinates) of the rink.
                "defense": The defensive half (smallest x-coordinates).
                "ozone": The offensive zone (blue line to end boards).
                "dzone": The defensive zone (end boards to blue line).

                Note that plot_range only affects what portion is plotted. Coordinates outside the range can still
                impact what is shown.

            plot_xlim: float or (float, float) (optional)
                The range of x-coordinates to include in the plot.
                    float: the lower bound of the x-coordinates. The upper bound will be the boards.
                    (float, float): The lower and upper bounds of the x-coordinates.

                Note that plot_xlim only affects what portion is plotted. Coordinates outside the range can still
                impact what is shown.

            plot_ylim: float or (float, float) (optional)
                The range of y-coordinates to include in the plot.
                    float: the lower bound of the y-coordinates. The upper bound will be the boards.
                    (float, float): The lower and upper bounds of the y-coordinates.

                Note that plot_ylim only affects what portion is plotted. Coordinates outside the range can still
                impact what is shown.

        Returns:
            plt.Polygon
        """

        if ax is None:
            ax = plt.gca()

        transform = self._get_transform(ax)
        x, y = self._boards.get_xy_for_clip()

        if not (plot_range is plot_xlim is plot_ylim is None):
            plot_range = plot_range or "full"
            plot_xlim, plot_ylim = self._get_limits(plot_range, plot_xlim, plot_ylim)
            x = np.clip(x, *plot_xlim)
            y = np.clip(y, *plot_ylim)

        return plt.Polygon(
            tuple(zip(x, y)),
            transform=transform,
        )

    def clip_plot(self, plot_features, ax=None, plot_range=None, plot_xlim=None, plot_ylim=None):
        """ Clip the provided plotted features to the boards subset to the desired plotting region.

        Arguments:
            plot_features: matplotlib object with a set_clip_path method.
                Typically, the returned objects from calling a matplotlib plotting method.

            ax: matplotlib Axes

            plot_range: {"full", "half", "offense", "defense", "ozone", "dzone"} (optional)
                Restricts the portion of the rink that can be plotted to beyond just the boards.

                Only affects x-coordinates and can be used in conjunction with ylim, but will be superceded by
                xlim if provided.

                "full": The entire length of the rink.
                "half" or "offense": The offensive half (largest x-coordinates) of the rink.
                "defense": The defensive half (smallest x-coordinates).
                "ozone": The offensive zone (blue line to end boards).
                "dzone": The defensive zone (end boards to blue line).

                Note that plot_range only affects what portion is plotted. Coordinates outside the range can still
                impact what is shown.

            plot_xlim: float or (float, float) (optional)
                The range of x-coordinates to include in the plot.
                    float: the lower bound of the x-coordinates. The upper bound will be the boards.
                    (float, float): The lower and upper bounds of the x-coordinates.

                Note that plot_xlim only affects what portion is plotted. Coordinates outside the range can still
                impact what is shown.

            plot_ylim: float or (float, float) (optional)
                The range of y-coordinates to include in the plot.
                    float: the lower bound of the y-coordinates. The upper bound will be the boards.
                    (float, float): The lower and upper bounds of the y-coordinates.

                Note that plot_ylim only affects what portion is plotted. Coordinates outside the range can still
                impact what is shown.
        """

        clip_path = self.get_clip_path(ax, plot_range, plot_xlim, plot_ylim)
        for plot_feature in np.ravel(plot_features):
            plot_feature.set_clip_path(clip_path)

    def _update_display_range(self, ax, **kwargs):
        """ Update the display range to include the outermost x,y-coordinate. """

        x_kws = [k for k in kwargs.keys() if re.fullmatch("x([0-9])*", k)]
        y_kws = [k for k in kwargs.keys() if re.fullmatch("y([0-9])*", k)]

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Don't want ax.transData included in transform when finding coordinates.
        transform = self.get_plot_transform(ax, kwargs.get("transform"), False)

        for x_kw, y_kw in zip(sorted(x_kws), sorted(y_kws)):
            xy = transform.transform(tuple(zip(kwargs[x_kw], kwargs[y_kw])))
            x, y = zip(*xy)

            xlim = [min(xlim[0], np.min(x)), max(xlim[1], np.max(x))]
            ylim = [min(ylim[0], np.min(y)), max(ylim[1], np.max(y))]

            ab = Affine2D().translate(-self.x_shift, -self.y_shift) + Affine2D().rotate_deg(self.rotation)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    def plot_fn(
        self,
        fn,
        clip_to_boards=True, update_display_range=False,
        plot_range=None, plot_xlim=None, plot_ylim=None,
        skip_draw=False, draw_kw=None,
        use_rink_coordinates=True,
        **kwargs
    ):
        """ Abstract plotting method. Can be used to call various matplotlib and seaborn plotting functions. Will
        attempt to apply appropriate transformations to the data based on the rink.

        Note that there may be functions for which this won't work/be appropriate. Also, all parameters passed to the
        plotting function are keyword only.

        Arguments:
            fn: a matplotlib or seaborn plotting function

            clip_to_boards: bool (default=True)
                Whether or not to clip the plot to stay within the bounds of the boards.

            update_display_range: bool (default=False)
                Whether or not to update the display range for plotted objects outside of the rink.

                The display range will be updated to the extremity of the passed in coordinates. If, for example, scatter
                points are outside the rink, half of the outermost point may be cut off.

                Adding Text will automatically update the display range, regardless of what is set here.

            plot_range: {"full", "half", "offense", "defense", "ozone", "dzone"} (optional)
                Restricts the portion of the rink that can be plotted to beyond just the boards.

                Only affects x-coordinates and can be used in conjunction with ylim, but will be superceded by
                xlim if provided.

                "full": The entire length of the rink.
                "half" or "offense": The offensive half (largest x-coordinates) of the rink.
                "defense": The defensive half (smallest x-coordinates).
                "ozone": The offensive zone (blue line to end boards).
                "dzone": The defensive zone (end boards to blue line).

                Note that plot_range only affects what portion is plotted. Coordinates outside the range can still
                impact what is shown.

            plot_xlim: float or (float, float) (optional)
                The range of x-coordinates to include in the plot.
                    float: the lower bound of the x-coordinates. The upper bound will be the boards.
                    (float, float): The lower and upper bounds of the x-coordinates.

                Note that plot_xlim only affects what portion is plotted. Coordinates outside the range can still
                impact what is shown.

            plot_ylim: float or (float, float) (optional)
                The range of y-coordinates to include in the plot.
                    float: the lower bound of the y-coordinates. The upper bound will be the boards.
                    (float, float): The lower and upper bounds of the y-coordinates.

                Note that plot_ylim only affects what portion is plotted. Coordinates outside the range can still
                impact what is shown.

            skip_draw: bool (default=False)
                If the rink has not already been drawn, setting to True will prevent the rink from being drawn.

            draw_kw: dict (optional)
                If the rink has not already been drawn, keyword arguments to pass to the draw method.

            use_rink_coordinates: bool (default=True)
                Whether or not the plotted features are using the rink's coordinates. If, eg, adding text relative the
                size of the figure instead, this should be set to False.

            kwargs: dict
                All parameters to be passed to the plotting function.

        Returns:
            The result from calling fn.
        """

        # Most plotting will be done with ax.* in which case the Axes object is included in fn.
        # In cases where it isn't (eg seaborn functions), need to find it in kwargs.
        try:
            is_ax_fn = isinstance(fn.__self__, Axes)
        except AttributeError:
            is_ax_fn = False

        ax = fn.__self__ if is_ax_fn else kwargs.pop("ax", plt.gca())

        # Draw rink if not already drawn.
        if not (ax in self._drawn or skip_draw):
            draw_kw = draw_kw or {}
            self.draw(ax=ax, **draw_kw)

        # Create boards constraint.
        if clip_to_boards and "clip_path" not in kwargs:
            kwargs["clip_path"] = self.get_clip_path(ax, plot_range, plot_xlim, plot_ylim)

        # Only use rink transform if plotting based on rink coordinates.
        if use_rink_coordinates:
            if update_display_range and not clip_to_boards:
                self._update_display_range(ax, **kwargs)

            # Update transform after display range to access original transform in _update_display_range.
            kwargs["transform"] = self.get_plot_transform(ax, kwargs.get("transform"))

        try:
            return fn(ax=ax, **kwargs)
        except (AttributeError, TypeError):
            return fn(**kwargs)
