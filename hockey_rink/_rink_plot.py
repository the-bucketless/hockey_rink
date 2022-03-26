""" Module for adding plotting functions to BaseRink.

Not intended for direct use, only as a parent class.
"""


from functools import wraps
from hockey_rink._base_rink import BaseRink
import matplotlib.pyplot as plt
import numpy as np
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
                values = np.ones(x.shape)
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
                plot_xlim, plot_ylim = self._get_limits(plot_range,
                                                        self.copy_(plot_xlim),
                                                        self.copy_(plot_ylim))

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

        constraint = self._add_boards_constraint(ax, transform)

        for plot_feature in plot_features:
            plot_feature.set_clip_path(constraint)

    def _outside_rink_to_nan(self, x, y, values):
        """ Set values of coordinates outside the boundaries of the rink to nan. """

        x = np.abs(x).astype("float32")
        y = np.abs(y).astype("float32")

        values = values.astype("float32")

        center_x = self._boards_constraint.length / 2 - self._boards_constraint.radius
        center_y = self._boards_constraint.width / 2 - self._boards_constraint.radius

        mask = ((x > center_x) & (y > center_y)
                & ((center_x - x) ** 2 + (center_y - y) ** 2 > self._boards_constraint.radius ** 2))
        values[mask] = np.nan

        return values

    def _update_display_range(self, x, y, ax):
        """ Update xlim and ylim for plotted features not constrained to the rink.

        Parameters:
            x: array_like

            y: array_like

            ax: matplotlib Axes
                Axes in which the features were plotted.
        """

        curr_xlim = ax.get_xlim()
        curr_ylim = ax.get_ylim()

        full_x = [*curr_xlim, *x]
        full_y = [*curr_ylim, *y]

        ax.set_xlim(min(full_x), max(full_x))
        ax.set_ylim(min(full_y), max(full_y))

    @staticmethod
    def binned_stat_2d(x, y, values, statistic="sum", xlim=None, ylim=None, binsize=1, bins=None):
        """ Use scipy to compute a bidimensional binned statistic.

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
    def plot(self, x, y, *, is_constrained=True, zorder=20, ax=None, **kwargs):
        """ Wrapper for matplotlib plot function.

        Will plot to areas out of view when full ice surface is not displayed.

        All parameters other than x and y require keywords.
            ie) plot(x, y, False) won't work, needs to be plot(x, y, is_constrained=False)

        Parameters:
            x: array_like

            y: array_like

            is_constrained: bool; default: True
                Indicates whether or not the plot is constrained to remain inside the boards.

            zorder: float; default: 20
                Determines which rink features the plot will draw over.

            ax: matplotlib Axes; optional
                Axes in which to draw the plot.  If not provided, will use the currently active Axes.

            **kwargs: Any other matplotlib plot properties; optional

        Returns:
            list of matplotlib Line2D
        """

        img = ax.plot(x, y, zorder=zorder, **kwargs)

        if is_constrained:
            self._constrain_plot(img, ax, kwargs["transform"])
        else:
            self._update_display_range(x, y, ax)

        return img

    @_validate_plot
    def scatter(self, x, y, *, is_constrained=True, symmetrize=False,
                zorder=20, ax=None, **kwargs):
        """ Wrapper for matplotlib scatter function.

        Will plot to areas out of view when full ice surface is not displayed.

        All parameters other than x and y require keywords.
            ie) scatter(x, y, False) won't work, needs to be scatter(x, y, is_constrained=False)

        Parameters:
            x: array_like

            y: array_like

            symmetrize: bool; default: False
                Indicates whether or not to reflect the coordinates across the y-axis.

            is_constrained: bool; default: True
                Indicates whether or not the plot is constrained to remain inside the boards.

            zorder: float; default: 20
                Determines which rink features the plot will draw over.

            ax: matplotlib Axes; optional
                Axes in which to draw the plot.  If not provided, will use the currently active Axes.

            **kwargs: Any other matplotlib scatter properties; optional

        Returns:
            matplotlib PathCollection
        """

        img = ax.scatter(x, y, zorder=zorder, **kwargs)

        if is_constrained:
            self._constrain_plot(img, ax, kwargs["transform"])
        else:
            self._update_display_range(x, y, ax)

        return img

    @_validate_plot
    def arrow(self, x1, y1, x2, y2, *, is_constrained=True,
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

        if is_constrained:
            self._constrain_plot(arrows, ax, kwargs["transform"])
        else:
            self._update_display_range([*x1, *x2], [*y1, *y2], ax)

        return arrows

    @_validate_plot
    @_validate_values
    def hexbin(self, x, y, *, values=None, is_constrained=True, symmetrize=False,
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

            symmetrize: bool; default: False
                Indicates whether or not to reflect the coordinates and values across the y-axis.

            is_constrained: bool; default: True
                Indicates whether or not the plot is constrained to remain inside the boards.

                If plot ranges are used, also constrains coordinates included to remain inside the boards.

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

        # setting clip_on to True can lead to odd results
        kwargs["clip_on"] = kwargs.get("clip_on", True)

        try:
            iter(binsize)
        except:
            binsize = (binsize, binsize)

        default_gridsize = (int((plot_xlim[1] - plot_xlim[0]) / binsize[0]),
                            int((plot_ylim[1] - plot_ylim[0]) / binsize[1]))
        gridsize = gridsize or default_gridsize

        img = ax.hexbin(x, y, C=values, gridsize=gridsize, zorder=zorder, **kwargs)

        if is_constrained:
            self._constrain_plot(img, ax, kwargs["transform"])
        else:
            self._update_display_range(x, y, ax)

        return img

    @_validate_plot
    @_validate_values
    def heatmap(self, x, y, *, values=None, symmetrize=False, is_constrained=True,
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

            symmetrize: bool; default: False
                Indicates whether or not to reflect the coordinates and values across the y-axis.

            is_constrained: bool; default: True
                Indicates whether or not the plot is constrained to remain inside the boards.

                If plot ranges are used, also constrains coordinates included to remain inside the boards.

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

        if is_constrained:
            self._constrain_plot(img, ax, kwargs["transform"])
        else:
            self._update_display_range(x, y, ax)

        return img

    @_validate_plot
    @_validate_values
    def contour(self, x, y, *, values=None, fill=True, symmetrize=False, is_constrained=True,
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

            symmetrize: bool; default: False
                Indicates whether or not to reflect the coordinates and values across the y-axis.

            is_constrained: bool; default: True
                Indicates whether or not the plot is constrained to remain inside the boards.

                If plot ranges are used, also constrains coordinates included to remain inside the boards.

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

        if fill:
            img = ax.contourf(x_centers, y_centers, stat, zorder=zorder, **kwargs)
        else:
            img = ax.contour(x_centers, y_centers, stat, zorder=zorder, **kwargs)

        if is_constrained:
            self._constrain_plot(img.collections, ax, kwargs["transform"])
        else:
            self._update_display_range(x, y, ax)

        return img

    """ Alias for contour. """
    contourf = contour
