""" Module for adding plotting functions to BaseRink.

Not intended for direct use, only as a parent class.
"""


from hockey_rink._base_rink import BaseRink
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np
import re


class BaseRinkPlot(BaseRink):
    """ Class extending BaseRink to include plotting methods. """

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

                Can be used in conjunction with plot_ylim. When used without plot_ylim, the y-coordinates will be the
                entire width of the ice. Will be superseded by plot_xlim if provided.

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
            plot_xlim, plot_ylim = self._get_limits(plot_range, plot_xlim, plot_ylim, False)
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

                Can be used in conjunction with plot_ylim. When used without plot_ylim, the y-coordinates will be the
                entire width of the ice. Will be superseded by plot_xlim if provided.

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
            xy = transform.transform(tuple(zip(np.ravel(kwargs[x_kw]), np.ravel(kwargs[y_kw]))))
            x, y = zip(*xy)

            xlim = [min(xlim[0], np.min(x)), max(xlim[1], np.max(x))]
            ylim = [min(ylim[0], np.min(y)), max(ylim[1], np.max(y))]

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    def binned_statistic_2d(
        self,
        x, y, values,
        nbins=10, binsize=None,
        reduce_fn=np.mean, fill_value=np.nan,
        as_center_bin=False,
        plot_range=None, plot_xlim=None, plot_ylim=None,
    ):
        """ Compute a bi-dimensional binned statistic.

        When creating centered bins, by default, two extra bins are created to enforce symmetry when flipping
        coordinates. Though, there may still be minor differences as a result of coordinate cutoffs (< vs <=).

        Parameters:
            x: array-like
            y: array-like

            values: array_like (optional)
                If None, values of 1 will be assigned to each coordinate.

            nbins: int or (int, int) (optional)
                The number of bins.

                The first int will be the number of bins used in the x direction and the second in the y. If only one
                value provided, it will be used for both.

                Will be ignored if binsize is not None. At least one of nbins and binsize must be provided.

            binsize: float or (float, float) (optional)
                The size of the bins.

                The first float will be the size of the bins in the x direction and the second in the y. If only one
                value provided, it will be used for both.

                At least one of nbins and binsize must be provided.

            reduce_fn: str or callable (default np.mean)
                The function used on the binned statistics.

                The following are available str options:
                    "mean"
                    "sum"
                    "count"
                    "std"
                    "median"
                    "min"
                    "max"

                If None, will default to count.

            fill_value: float (default=np.nan)
                The value used when no values are present in a coordinate bin.

            as_center_bin: bool (default=False)
                Whether or not to use the center of bins rather than the endpoints.

            plot_range: {"full", "half", "offense", "defense", "ozone", "dzone"} (optional)
                Restricts the portion of the rink that values can be calculated from.

                Can be used in conjunction with plot_ylim. When used without plot_ylim, the y-coordinates will be the
                entire width of the ice. Will be superseded by plot_xlim if provided.

                "full": The entire length of the rink.
                "half" or "offense": The offensive half (largest x-coordinates) of the rink.
                "defense": The defensive half (smallest x-coordinates).
                "ozone": The offensive zone (blue line to end boards).
                "dzone": The defensive zone (end boards to blue line).

            plot_xlim: float or (float, float) (optional)
                The range of x-coordinates to calculate values from.
                    float: the lower bound of the x-coordinates. The upper bound will be the boards.
                    (float, float): The lower and upper bounds of the x-coordinates.

                Note that plot_xlim only affects what portion is plotted. Coordinates outside the range can still
                impact what is shown.

            plot_ylim: float or (float, float) (optional)
                The range of y-coordinates to calculate values from.
                    float: the lower bound of the y-coordinates. The upper bound will be the boards.
                    (float, float): The lower and upper bounds of the y-coordinates.

                Note that plot_ylim only affects what portion is plotted. Coordinates outside the range can still
                impact what is shown.
        """

        values = np.ones_like(x) if values is None else np.array(values)

        reduce_fn_names = {
            None: np.ma.count,
            "mean": np.mean,
            "sum": np.sum,
            "count": np.ma.count,
            "std": np.std,
            "median": np.median,
            "min": np.min,
            "max": np.max,
        }

        try:
            reduce_fn = reduce_fn.lower()
        except AttributeError:
            pass

        reduce_fn = reduce_fn_names.get(reduce_fn, reduce_fn)

        if (plot_range, plot_xlim, plot_ylim) != (None, None, None):
            xlim, ylim = self._get_limits(plot_range, plot_xlim, plot_ylim, False)

            # Need to shift to allow for reverse shift when plotting.
            xlim = [x + self.x_shift for x in xlim]
            ylim = [y + self.y_shift for y in ylim]

        # Use the data to set anything not provided.
        if plot_range is plot_xlim is None:
            xlim = [np.min(x), np.max(x)]
        if plot_range is plot_ylim is None:
            ylim = [np.min(y), np.max(y)]

        if binsize is None:
            try:
                iter(nbins)
            except TypeError:
                nbins = [nbins]

            xbins = np.linspace(*xlim, nbins[0] + 1)
            ybins = np.linspace(*ylim, nbins[-1] + 1)
        else:
            try:
                iter(binsize)
            except TypeError:
                binsize = [binsize]

            xbins = np.arange(xlim[0], xlim[1] + binsize[0], binsize[0])
            ybins = np.arange(ylim[0], ylim[1] + binsize[-1], binsize[-1])

        # Add an extra bins to the start and end when centering.
        delta_x = xbins[1] - xbins[0]
        delta_y = ybins[1] - ybins[0]
        if as_center_bin:
            xbins = np.concatenate([xbins[0] - delta_x, xbins, xbins[-1] + delta_x], axis=None)
            ybins = np.concatenate([ybins[0] - delta_y, ybins, ybins[-1] + delta_y], axis=None)

        n_xbins = len(xbins) - 1
        n_ybins = len(ybins) - 1

        # By default, side="left" which forces values from lower bound into bin by themselves. Clip to push them into
        # 1st bin instead of 0th and subtract 1 to get correct number of bins.
        x_binnumbers = np.clip(np.searchsorted(xbins, x), 1, None) - 1
        y_binnumbers = np.clip(np.searchsorted(ybins, y), 1, None) - 1

        # matplotlib functions are y, x instead of x, y.
        result = np.full((n_ybins, n_xbins), fill_value, dtype="float")
        for i in range(n_xbins):
            for j in range(n_ybins):
                group_vals = values[(x_binnumbers == i) & (y_binnumbers == j)]

                if group_vals.size:
                    result[j, i] = reduce_fn(group_vals)

        if as_center_bin:
            xbins = (xbins[1:] + xbins[:-1]) / 2
            ybins = (ybins[1:] + ybins[:-1]) / 2

        return result, xbins, ybins

    def _process_plot(
        self,
        ax,
        clip_to_boards=True, update_display_range=False,
        plot_range=None, plot_xlim=None, plot_ylim=None,
        skip_draw=False, draw_kw=None,
        use_rink_coordinates=True,
        nbins=None, binsize=None,
        reduce_fn=np.mean, fill_value=np.nan,
        as_center_bin=False,
        zorder=20,
        **kwargs,
    ):
        """ Perform preprocessing steps before plotting. """

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

        # Update zorder.
        kwargs["zorder"] = kwargs.get("zorder", zorder)

        # Find bins and statistic for plots requiring it.
        if nbins is not None or binsize is not None:
            x = kwargs.pop("x")
            y = kwargs.pop("y")
            values = kwargs.pop("values", np.ones_like(x))
            stat, x_edge, y_edge = self.binned_statistic_2d(
                x, y, values,
                nbins, binsize,
                reduce_fn, fill_value,
                as_center_bin,
                plot_range, plot_xlim, plot_ylim,
            )
            kwargs["x"] = x_edge
            kwargs["y"] = y_edge
            kwargs["values"] = stat

        return kwargs

    def plot_fn(
        self,
        fn,
        clip_to_boards=True, update_display_range=False,
        plot_range=None, plot_xlim=None, plot_ylim=None,
        skip_draw=False, draw_kw=None,
        use_rink_coordinates=True,
        nbins=None, binsize=None,
        reduce_fn=np.mean, fill_value=np.nan,
        as_center_bin=False,
        zorder=20,
        position_args=None,
        **kwargs
    ):
        """ Wrapper method to be used to call various matplotlib and seaborn plotting functions. Will attempt to apply
        appropriate transformations to the data based on the rink.

        Note that there may be functions for which this won't work/be appropriate. Also, all parameters passed to the
        plotting function are keyword only.

        Arguments:
            fn: a matplotlib or seaborn plotting function

            clip_to_boards: bool (default=True)
                Whether or not to clip the plot to stay within the bounds of the boards.

            update_display_range: bool (default=False)
                Whether or not to update the display range for plotted objects outside of the rink.

                The display range will be updated to the extremity of the passed in coordinates. If, for example,
                scatter points are outside the rink, half of the outermost point may be cut off.

                Adding Text will automatically update the display range, regardless of what is set here.

            plot_range: {"full", "half", "offense", "defense", "ozone", "dzone"} (optional)
                Restricts the portion of the rink that can be plotted to beyond just the boards.

                Can be used in conjunction with plot_ylim. When used without plot_ylim, the y-coordinates will be the
                entire width of the ice. Will be superseded by plot_xlim if provided.

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

            nbins: int or (int, int) (optional)
                The number of bins in plots requiring binned statistics (eg heatmap, contour).

                The first int will be the number of bins used in the x direction and the second in the y. If only one
                value provided, it will be used for both.

                Will be ignored if binsize is not None. At least one of nbins and binsize must be provided for plots
                that required binned statistics.

            binsize: float or (float, float) (optional)
                The size of bins in plots requiring binned statistics (eg heatmap, contour).

                The first float will be the size of the bins in the x direction and the second in the y. If only one
                value provided, it will be used for both.

                At least one of nbins and binsize must be provided for plots that require binned statistics.

            reduce_fn: str or callable (default np.mean)
                The function to use on binned statistics.

                The following are available str options:
                    "mean"
                    "sum"
                    "count"
                    "std"
                    "median"
                    "min"
                    "max"

                If None, will default to count.

            fill_value: float (default=np.nan)
                The value used in plots requiring binned statistics when no values are present in a coordinate bin.

            as_center_bin: bool (default=False)
                Whether or not to use the center of bins for plots requiring binned statistics rather than the
                endpoints. Used in contour plots but not heatmaps.

            zorder: float (default=20)
                Determines which rink features the plot will draw over.

            position_args: list (optional)
                Parameters that can't be passed to matplotlib functions as keywords.
                For example, ax.plot(x=x, y=y) will result in an error. To avoid this, set position_args to ["x", "y"].

            kwargs: All parameters to be passed to the plotting function.

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

        kwargs = self._process_plot(
            ax,
            clip_to_boards, update_display_range,
            plot_range, plot_xlim, plot_ylim,
            skip_draw, draw_kw,
            use_rink_coordinates,
            nbins, binsize,
            reduce_fn, fill_value,
            as_center_bin,
            zorder,
            **kwargs,
        )

        position_args = position_args or []
        args = [kwargs.pop(arg) for arg in position_args]

        if is_ax_fn:
            return fn(*args, **kwargs)
        else:
            return fn(*args, **kwargs, ax=ax)

    def scatter(
        self,
        x, y,
        ax=None,
        clip_to_boards=True, update_display_range=False,
        plot_range=None, plot_xlim=None, plot_ylim=None,
        skip_draw=False, draw_kw=None,
        use_rink_coordinates=True,
        **kwargs
    ):
        """ Wrapper for matplotlib scatter function.

        Parameters:
            x: array-like or key in data
            y: array-like or key in data

            ax: matplotlib Axes (optional)
                If not provided, will use the currently active Axes.

            clip_to_boards: bool (default=True)
                Whether or not to clip the plot to stay within the bounds of the boards.

            update_display_range: bool (default=False)
                Whether or not to update the display range for plotted objects outside of the rink.

                The display range will be updated to the extremity of the passed in coordinates. If, for example,
                scatter points are outside the rink, half of the outermost point may be cut off.

            plot_range: {"full", "half", "offense", "defense", "ozone", "dzone"} (optional)
                Restricts the portion of the rink that can be plotted to beyond just the boards.

                Can be used in conjunction with plot_ylim. When used without plot_ylim, the y-coordinates will be the
                entire width of the ice. Will be superseded by plot_xlim if provided.

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
                Whether or not the plotted features are using the rink's coordinates.

            kwargs: Any other matplotlib scatter properties. (optional)

        Returns:
            matplotlib PathCollection
        """

        if ax is None:
            ax = plt.gca()

        return self.plot_fn(
            ax.scatter,
            clip_to_boards, update_display_range,
            plot_range, plot_xlim, plot_ylim,
            skip_draw, draw_kw,
            use_rink_coordinates,
            x=x, y=y,
            **kwargs,
        )

    def plot(
        self,
        x, y, fmt=None,
        ax=None,
        clip_to_boards=True, update_display_range=False,
        plot_range=None, plot_xlim=None, plot_ylim=None,
        skip_draw=False, draw_kw=None,
        use_rink_coordinates=True,
        **kwargs
    ):
        """ Wrapper for matplotlib plot function.

        Parameters:
            x: array-like or key in data
            y: array-like or key in data

            fmt: str (optional)
                A format string. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html for details.

            ax: matplotlib Axes (optional)
                If not provided, will use the currently active Axes.

            clip_to_boards: bool (default=True)
                Whether or not to clip the plot to stay within the bounds of the boards.

            update_display_range: bool (default=False)
                Whether or not to update the display range for plotted objects outside of the rink.

                The display range will be updated to the extremity of the passed in coordinates. If, for example,
                scatter points are outside the rink, half of the outermost point may be cut off.

            plot_range: {"full", "half", "offense", "defense", "ozone", "dzone"} (optional)
                Restricts the portion of the rink that can be plotted to beyond just the boards.

                Can be used in conjunction with plot_ylim. When used without plot_ylim, the y-coordinates will be the
                entire width of the ice. Will be superseded by plot_xlim if provided.

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
                Whether or not the plotted features are using the rink's coordinates.

            kwargs: Any other matplotlib plot properties. (optional)

        Returns:
            list of matplotlib Line2D
        """

        if ax is None:
            ax = plt.gca()

        position_args = ["x", "y"]
        if fmt is not None:
            kwargs["fmt"] = fmt
            position_args.append("fmt")

        return self.plot_fn(
            ax.plot,
            clip_to_boards, update_display_range,
            plot_range, plot_xlim, plot_ylim,
            skip_draw, draw_kw,
            use_rink_coordinates,
            position_args=position_args,
            x=x, y=y,
            **kwargs,
        )

    def arrow(
        self,
        x, y,
        dx=None, dy=None,
        x2=None, y2=None,
        ax=None,
        clip_to_boards=True, update_display_range=False,
        plot_range=None, plot_xlim=None, plot_ylim=None,
        skip_draw=False, draw_kw=None,
        use_rink_coordinates=True,
        **kwargs
    ):
        """ Wrapper for matplotlib arrow function.

        Allows for arrow endpoints to be recorded as either delta values (dx, dy) or coordinates (x2, y2).

        Parameters:
            x: array-like or key in data
                The x-coordinates of the base of the arrows.

            y: array-like or key in data
                The y-coordinates of the base of the arrows.

            dx: array-like or key in data (optional)
                The length of the arrow in the x direction.
                One of dx and x2 has to be specified.

            dy: array-like or key in data (optional)
                The length of the arrow in the y direction.
                One of dy and y2 has to be specified.

            x2: array-like or key in data (optional)
                The endpoint of the arrow.
                One of dx and x2 has to be specified.

            y2: array-like or key in data (optional)
                The endpoint of the arrow.
                One of dy and y2 has to be specified.

            ax: matplotlib Axes (optional)
                If not provided, will use the currently active Axes.

            clip_to_boards: bool (default=True)
                Whether or not to clip the plot to stay within the bounds of the boards.

            update_display_range: bool (default=False)
                Whether or not to update the display range for plotted objects outside of the rink.

                The display range will be updated to the extremity of the passed in coordinates. If, for example,
                arrows are outside the rink, the head may be cut off.

            plot_range: {"full", "half", "offense", "defense", "ozone", "dzone"} (optional)
                Restricts the portion of the rink that can be plotted to beyond just the boards.

                Can be used in conjunction with plot_ylim. When used without plot_ylim, the y-coordinates will be the
                entire width of the ice. Will be superseded by plot_xlim if provided.

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
                Whether or not the plotted features are using the rink's coordinates.

            kwargs: Any other matplotlib arrow properties. (optional)

        Returns:
            list of matplotlib FancyArrow
        """

        if ax is None:
            ax = plt.gca()

        data = kwargs.pop("data", None)

        x, y, dx, dy, x2, y2 = [
            data[var] if isinstance(var, str) else var
            for var in (x, y, dx, dy, x2, y2)
        ]

        x = np.ravel(x)
        y = np.ravel(y)

        # Calculate the length of the arrow if not provided.
        if dx is None:
            dx = x2 - x
        if dy is None:
            dy = y2 - y

        dx = np.ravel(dx)
        dy = np.ravel(dy)

        return [
            self.plot_fn(
                ax.arrow,
                clip_to_boards, update_display_range,
                plot_range, plot_xlim, plot_ylim,
                skip_draw, draw_kw,
                use_rink_coordinates,
                x=x_, y=y_,
                dx=dx_, dy=dy_,
                **kwargs,
            )
            for x_, y_, dx_, dy_ in zip(x, y, dx, dy)
        ]

    def hexbin(
        self,
        x, y, values=None,
        ax=None,
        clip_to_boards=True, update_display_range=False,
        plot_range=None, plot_xlim=None, plot_ylim=None,
        skip_draw=False, draw_kw=None,
        use_rink_coordinates=True,
        zorder=3,
        **kwargs
    ):
        """ Wrapper for matplotlib hexbin function.

        Parameters:
            x: array-like or key in data
            y: array-like or key in data

            values: array_like or key in data (optional)
                If None, values of 1 will be assigned to each coordinate.

            ax: matplotlib Axes (optional)
                If not provided, will use the currently active Axes.

            clip_to_boards: bool (default=True)
                Whether or not to clip the plot to stay within the bounds of the boards.

            update_display_range: bool (default=False)
                Whether or not to update the display range for plotted objects outside of the rink.

                The display range will be updated to the extremity of the passed in coordinates. If, for example,
                hexagons are outside the rink, half of the outermost hexagon may be cut off.

            plot_range: {"full", "half", "offense", "defense", "ozone", "dzone"} (optional)
                Restricts the portion of the rink that can be plotted to beyond just the boards.

                Can be used in conjunction with plot_ylim. When used without plot_ylim, the y-coordinates will be the
                entire width of the ice. Will be superseded by plot_xlim if provided.

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
                Whether or not the plotted features are using the rink's coordinates.

            zorder: float (default=3)
                Determines which rink features the plot will draw over.

            kwargs: Any other matplotlib hexbin properties. (optional)

        Returns:
            matplotlib PolyCollection
        """

        if ax is None:
            ax = plt.gca()

        values = kwargs.get("C", values)

        # Default to sum instead of mean when values not provided.
        if values is None:
            kwargs["reduce_C_function"] = kwargs.get("reduce_C_function", np.sum)

        transform = self.get_plot_transform(ax, kwargs.get("transform"), False)
        rotation = self._rotations.get(ax, Affine2D().rotate_deg(self.rotation))

        # Newer versions of hexbin don't rotate the position of the hexagons.
        # Need to delay application of transform until after drawing hexbin.
        kwargs = self._process_plot(
            ax,
            clip_to_boards, update_display_range,
            plot_range, plot_xlim, plot_ylim,
            skip_draw, draw_kw,
            use_rink_coordinates,
            zorder=zorder,
            x=x, y=y, C=values,
            **kwargs,
        )

        kwargs.pop("transform")

        img = ax.hexbin(**kwargs)

        # Rotate vertices and transform offsets.
        hexagon = img.get_paths()[0]
        hexagon.vertices = rotation.transform(hexagon.vertices)
        img.set_offsets(transform.transform(img.get_offsets()))

        return img

    def heatmap(
        self,
        x, y, values=None,
        nbins=10, binsize=None,
        reduce_fn=np.mean, fill_value=np.nan,
        ax=None,
        clip_to_boards=True, update_display_range=False,
        plot_range=None, plot_xlim=None, plot_ylim=None,
        skip_draw=False, draw_kw=None,
        use_rink_coordinates=True,
        zorder=3,
        **kwargs
    ):
        """ Wrapper for matplotlib pcolormesh function.

        It's generally best to include a plot_range to ensure the area is accessed.

        Will attempt to create the bins based on the parameters. If the bins have already been set, this can be
        skipped by setting nbins to None.
            ie) rink.heatmap(xbins, ybins, values, nbins=None)

        Only x, y, and values will be looked for in the data parameter, if provided.

        Parameters:
            x: array-like or key in data
            y: array-like or key in data

            values: array_like or key in data (optional)
                If None, values of 1 will be assigned to each coordinate.

            nbins: int or (int, int) (optional)
                The number of bins.

                The first int will be the number of bins used in the x direction and the second in the y. If only one
                value provided, it will be used for both.

                Will be ignored if binsize is not None. At least one of nbins and binsize must be provided unless
                x and y are the bins to be used.

            binsize: float or (float, float) (optional)
                The size of the bins.

                The first float will be the size of the bins in the x direction and the second in the y. If only one
                value provided, it will be used for both.

                At least one of nbins and binsize must be provided unless x and y are the bins to be used.

            reduce_fn: str or callable (default np.mean)
                The function used on the binned statistics.

                The following are available str options:
                    "mean"
                    "sum"
                    "count"
                    "std"
                    "median"
                    "min"
                    "max"

                If None, will default to count.

            fill_value: float (default=np.nan)
                The value used when no values are present in a coordinate bin.

            ax: matplotlib Axes (optional)
                If not provided, will use the currently active Axes.

            clip_to_boards: bool (default=True)
                Whether or not to clip the plot to stay within the bounds of the boards.

            update_display_range: bool (default=False)
                Whether or not to update the display range for plotted objects outside of the rink.

                The display range will be updated to the extremity of the passed in coordinates. If, for example,
                heatmap pixels are outside the rink, half of the outermost pixel may be cut off.

            plot_range: {"full", "half", "offense", "defense", "ozone", "dzone"} (optional)
                Restricts the portion of the rink that can be plotted to beyond just the boards.

                Can be used in conjunction with plot_ylim. When used without plot_ylim, the y-coordinates will be the
                entire width of the ice. Will be superseded by plot_xlim if provided.

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
                Whether or not the plotted features are using the rink's coordinates.

            zorder: float (default=3)
                Determines which rink features the plot will draw over.

            kwargs: Any other matplotlib pcolormesh properties. (optional)

        Returns:
            matplotlib QuadMesh
        """

        if ax is None:
            ax = plt.gca()

        data = kwargs.pop("data", None)
        x, y, values = [data[var] if isinstance(var, str) else var for var in (x, y, values)]

        return self.plot_fn(
            ax.pcolormesh,
            clip_to_boards, update_display_range,
            plot_range, plot_xlim, plot_ylim,
            skip_draw, draw_kw,
            use_rink_coordinates,
            x=x, y=y,
            values=values,
            nbins=nbins, binsize=binsize,
            reduce_fn=reduce_fn, fill_value=fill_value,
            as_center_bin=False,
            zorder=zorder,
            position_args=["x", "y", "values"],
            **kwargs,
        )

    def contour(
        self,
        x, y, values=None,
        nbins=10, binsize=None,
        reduce_fn=np.mean, fill_value=0,
        ax=None,
        clip_to_boards=True, update_display_range=False,
        plot_range=None, plot_xlim=None, plot_ylim=None,
        skip_draw=False, draw_kw=None,
        use_rink_coordinates=True,
        zorder=3,
        fill=False,
        **kwargs
    ):
        """ Wrapper for matplotlib contour and contourf functions.

        contourf can be accessed by setting fill to True.

        It's generally best to include a plot_range to ensure the area is accessed.

        Will attempt to create the bins based on the parameters. If the bins have already been set, this can be
        skipped by setting nbins to None.
            ie) rink.contour(xbins, ybins, values, nbins=None)

        By default, two extra bins are created to enforce symmetry when flipping coordinates. Though, there may still
        be minor differences as a result of coordinate cutoffs (< vs <=).

        Only x, y, and values will be looked for in the data parameter, if provided.

        Parameters:
            x: array-like or key in data
            y: array-like or key in data

            values: array_like or key in data (optional)
                If None, values of 1 will be assigned to each coordinate.

            nbins: int or (int, int) (optional)
                The number of bins.

                The first int will be the number of bins used in the x direction and the second in the y. If only one
                value provided, it will be used for both.

                Will be ignored if binsize is not None. At least one of nbins and binsize must be provided unless
                x and y are the bins to be used.

            binsize: float or (float, float) (optional)
                The size of the bins.

                The first float will be the size of the bins in the x direction and the second in the y. If only one
                value provided, it will be used for both.

                At least one of nbins and binsize must be provided unless x and y are the bins to be used.

            reduce_fn: str or callable (default np.mean)
                The function used on the binned statistics.

                The following are available str options:
                    "mean"
                    "sum"
                    "count"
                    "std"
                    "median"
                    "min"
                    "max"

                If None, will default to count.

            fill_value: float (default=0)
                The value used when no values are present in a coordinate bin.

            ax: matplotlib Axes (optional)
                If not provided, will use the currently active Axes.

            clip_to_boards: bool (default=True)
                Whether or not to clip the plot to stay within the bounds of the boards.

            update_display_range: bool (default=False)
                Whether or not to update the display range for plotted objects outside of the rink.

                The display range will be updated to the extremity of the passed in coordinates.

            plot_range: {"full", "half", "offense", "defense", "ozone", "dzone"} (optional)
                Restricts the portion of the rink that can be plotted to beyond just the boards.

                Can be used in conjunction with plot_ylim. When used without plot_ylim, the y-coordinates will be the
                entire width of the ice. Will be superseded by plot_xlim if provided.

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
                Whether or not the plotted features are using the rink's coordinates.

            zorder: float (default=3)
                Determines which rink features the plot will draw over.

            fill: bool (default=False)
                Whether or not to fill in the contours (use contourf rather than contour).

            kwargs: Any other matplotlib contour properties. (optional)

        Returns:
            matplotlib QuadContourSet
        """

        if ax is None:
            ax = plt.gca()

        data = kwargs.pop("data", None)
        x, y, values = [data[var] if isinstance(var, str) else var for var in (x, y, values)]

        # ax.contour doesn't accept a clip_path parameter.
        # Need to delay clipping until after drawing.
        kwargs = self._process_plot(
            ax,
            False, update_display_range,
            plot_range, plot_xlim, plot_ylim,
            skip_draw, draw_kw,
            use_rink_coordinates,
            nbins, binsize,
            reduce_fn, fill_value,
            True,
            zorder,
            x=x, y=y, values=values,
            **kwargs,
        )

        contour_plot = ax.contourf if fill else ax.contour

        img = contour_plot(
            kwargs.pop("x"), kwargs.pop("y"), kwargs.pop("values"),
            **kwargs,
        )

        if clip_to_boards:
            self.clip_plot(img.collections, ax, plot_range, plot_xlim, plot_ylim)

        return img

    def contourf(self, *args, **kwargs):
        kwargs["fill"] = kwargs.pop("fill", True)
        return self.contour(*args, **kwargs)

    def text(
        self,
        x, y, s,
        ax=None,
        skip_draw=False, draw_kw=None,
        use_rink_coordinates=True,
        zorder=101,
        **kwargs
    ):
        """ Wrapper for matplotlib text function.

        Accepts multiple values for x, y, and s.
        Does not get clipped to boards and will update display range.
        Only x, y, and s will be looked for in the data parameter, if provided.

        Parameters:
            x: array-like or key in data
            y: array-like or key in data

            s: array-like or key in data
                The text. Multiple texts can be included with multiple coordinates.

            ax: matplotlib Axes (optional)
                If not provided, will use the currently active Axes.

            skip_draw: bool (default=False)
                If the rink has not already been drawn, setting to True will prevent the rink from being drawn.

            draw_kw: dict (optional)
                If the rink has not already been drawn, keyword arguments to pass to the draw method.

            use_rink_coordinates: bool (default=True)
                Whether or not the plotted features are using the rink's coordinates.

            zorder: float (default=101)
                Determines which rink features the text will be drawn over.

            kwargs: Any other matplotlib text properties. (optional)

        Returns:
            list of matplotlib Text
        """

        if ax is None:
            ax = plt.gca()

        data = kwargs.pop("data", None)
        if data is not None:
            if isinstance(x, str):
                x = data[x]
            if isinstance(y, str):
                y = data[y]
            if isinstance(s, str):
                try:
                    s = data[s]
                except KeyError:
                    pass

        # Set default transform if not using rink coordinates.
        if not use_rink_coordinates:
            kwargs["transform"] = kwargs.get("transform", ax.transAxes)

        return [
            self.plot_fn(
                ax.text,
                False, False,
                None, None, None,
                skip_draw, draw_kw,
                use_rink_coordinates,
                zorder=zorder,
                x=x_, y=y_, s=s_,
                **kwargs,
            )
            for x_, y_, s_ in zip(np.ravel(x), np.ravel(y), np.ravel(s))
        ]
