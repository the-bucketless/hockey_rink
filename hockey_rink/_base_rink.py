""" Abstract base class for drawing a hockey rink.

Not intended for direct use, only as a parent class.
"""


from abc import ABC
from hockey_rink.rink_features import Boards
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np


class BaseRink(ABC):
    """ Abstract base class for drawing rinks using matplotlib.

    Attributes:
        x_shift: float
            Amount x-coordinates are to be shifted.

            When viewing the rink horizontally, the coordinate of the center of the ice surface from left to right.
                eg) If using data with a coordinate system that goes from 0 to 200, x_shift should be 100.

            The actual coordinates won't be affected.  The purpose is to update the coordinates passed in to
            align with the drawing, not to alter the drawing to align with the coordinates.

        y_shift: float
            Amount y-coordinates are to be shifted.

            When viewing the rink horizontally, the coordinate of the center of the ice surface from bottom to top.
                eg) If using data with a coordinate system that goes from 0 to 85, y_shift should be 42.5.

            The actual coordinates won't be affected.  The purpose is to update the coordinates passed in to
            align with the drawing, not to alter the drawing to align with the coordinates.
    """

    def __init__(self, rotation=0, x_shift=0, y_shift=0, alpha=None, boards=None):
        """ Initializes attributes.

        Parameters:
            rotation: float (default=0)
                Degree to rotate the rink.

            x_shift: float (default=0)
                Amount x-coordinates are to be shifted.

                When viewing the rink horizontally, the coordinate of the center of the ice surface from left to right.
                    eg) If using data with a coordinate system that goes from 0 to 200, x_shift should be 100.

                The actual coordinates won't be affected.  The purpose is to update the coordinates passed in to
                align with the drawing, not to alter the drawing to align with the coordinates.

            y_shift: float (default=0)
                Amount y-coordinates are to be shifted.

                When viewing the rink horizontally, the coordinate of the center of the ice surface from bottom to top.
                    eg) If using data with a coordinate system that goes from 0 to 85, y_shift should be 42.5.

                The actual coordinates won't be affected.  The purpose is to update the coordinates passed in to
                align with the drawing, not to alter the drawing to align with the coordinates.

            alpha: float (default=0)
                The alpha blending value, between 0 (transparent) and 1 (opaque).

                If not None, will be used for all features of the rink that don't override it.

            boards: dict (optional)
                Attributes for the Boards object.
        """

        self.x_shift = x_shift
        self.y_shift = y_shift
        self._rotation = Affine2D().rotate_deg(rotation)
        boards = boards or {}
        self._boards = Boards(alpha=alpha, **boards)
        self._features = {}
        self._feature_xlim, self._feature_ylim = self._boards.get_limits()

    def _initialize_feature(self, feature_name, params, alpha):
        """ Initialize a feature of the rink at each coordinate it's required.

        Parameters:
            feature_name: str
                Name of the feature being initialized. Used as the key in the _features dict. If multiple features
                are being created, all features after the first will include an underscore and number following the
                feature name.

            params: dict
                The class of the feature and all parameters required to instantiate its class.

            alpha: float
                Universal alpha parameter to set transparency of all features that don't override it.
        """

        feature_class = params.pop("class")

        if params.pop("is_constrained", True):
            params["clip_xy"] = self._boards.get_xy_for_clip()

        xs = np.ravel(params.get("x", [0]))
        ys = np.ravel(params.get("y", [0]))

        x_reflections = [False, True] if params.pop("reflect_x", False) else [False]
        y_reflections = [False, True] if params.pop("reflect_y", False) else [False]

        params["alpha"] = params.get("alpha", alpha)

        for i, (x, y, x_reflection, y_reflection) in enumerate(
            product(xs, ys, x_reflections, y_reflections)
        ):
            feature_params = dict(params)

            feature_params["x"] = x * (-1 if x_reflection else 1)
            feature_params["y"] = y * (-1 if y_reflection else 1)
            feature_params["is_reflected_x"] = x_reflection
            feature_params["is_reflected_y"] = y_reflection

            numeral = f"_{i}" if i else ""

            self._features[f"{feature_name}{numeral}"] = feature_class(**feature_params)

    def _get_limits(self, display_range="full", xlim=None, ylim=None):
        """ Return the xlim and ylim values corresponding to the parameters.

        Doesn't allow for coordinates extending beyond the extent of the boards.
        """

        # if boards are included in the limits, need to include their thickness
        half_length = self._boards.length / 2 + self._boards.thickness

        # If nzone exists, its length may be needed to calculate xlim.
        try:
            half_nzone_length = self._features.get("nzone").length / 2
        except AttributeError:
            half_nzone_length = 0

        # Blue line is needed to calculate xlim when display_range is "nzone".
        nzone_xmin, nzone_xmax = np.inf, -np.inf
        for feature in ("nzone", "blue_line", "blue_line_1"):
            try:
                feature_x, _ = self._features[feature].get_polygon_xy()
                feature_clip_x, _ = self._features[feature].clip_xy
                nzone_xmin = min(nzone_xmin, np.min(feature_x), np.max(feature_clip_x))
                nzone_xmax = max(nzone_xmax, np.max(feature_x), np.min(feature_clip_x))
            except (AttributeError, KeyError):
                pass
            except TypeError:
                nzone_xmin = min(nzone_xmin, np.min(feature_x))
                nzone_xmax = max(nzone_xmax, np.max(feature_x))

        if xlim is None:
            equivalencies = {
                "half": "offense",
                "offence": "offense",
                "defence": "defense",
            }
            display_range = display_range.lower() \
                .replace(" ", "")
            display_range = equivalencies.get(display_range, display_range)

            xlims = {
                "offense": (0, half_length),
                "defense": (-half_length, 0),
                "ozone": (half_nzone_length, half_length),
                "nzone": (nzone_xmin, nzone_xmax),
                "dzone": (-half_length, -half_nzone_length),
            }

            xlim = xlims.get(display_range, (-half_length, half_length))
        else:
            try:
                xlim = (xlim[0] - self.x_shift, xlim[1] - self.x_shift)
            except TypeError:
                xlim -= self.x_shift

                if xlim >= half_length:
                    xlim = -half_length

                xlim = (xlim, half_length)

        half_width = self._boards.width / 2 + self._boards.thickness
        if ylim is None:
            ylim = (-half_width, half_width)
        else:
            try:
                ylim = (ylim[0] - self.y_shift, ylim[1] - self.y_shift)
            except TypeError:
                ylim -= self.y_shift

                if ylim >= half_width:
                    ylim = -half_width

                ylim = (ylim, half_width)

        # always have the smaller coordinate first
        if xlim[0] > xlim[1]:
            xlim = (xlim[1], xlim[0])
        if ylim[0] > ylim[1]:
            ylim = (ylim[1], ylim[0])

        # disallow coordinates from extending beyond the rink
        xlim = (max(xlim[0], -half_length), min(xlim[1], half_length))
        ylim = (max(ylim[0], -half_width), min(ylim[1], half_width))

        return xlim, ylim

    def _get_transform(self, ax):
        """ Return the matplotlib Transform to apply to features of the rink. """

        return self._rotation + ax.transData

    @staticmethod
    def copy_(param):
        """ Return a copy of a parameter where possible. """

        try:
            param = param.copy()
        except AttributeError:
            pass

        return param

    def _rotate_xy(self, x, y):
        """ Rotate x,y-coordinates with rink rotation. """
        if self._rotation:
            xy = self._rotation.transform(tuple(zip(x, y)))
            return xy[:, 0], xy[:, 1]
        else:
            return x, y

    def convert_xy(self, x, y):
        """ Convert x,y-coordinates to the scale used for the rink. """

        x = self.copy_(x)
        y = self.copy_(y)

        x = np.ravel(x) - self.x_shift
        y = np.ravel(y) - self.y_shift

        return self._rotate_xy(x, y)

    def draw(self, ax=None, display_range="full", xlim=None, ylim=None, rotation=None):
        """ Draw the rink.

        Parameters:
            ax: matplotlib Axes; optional
                Axes in which to draw the rink.  If not provided, the currently-active Axes is used.

            display_range: {"full", "half", "offense", "defense", "ozone", "dzone"}; default: "full"
                The portion of the rink to display.  The entire rink is drawn regardless, display_range only
                affects what is shown.

                Only affects x-coordinates and can be used in conjuction with ylim, but will be superceded by
                xlim if provided.

                If a rotation other than those resulting in the rink being drawn horizontal/vertical, coordinates
                outside of the display range may be included.

                Features not constrained within the boards (not including the boards) will only be displayed if
                display_range is "full".

                "full": The entire length of the rink is displayed.
                "half" or "offense": The offensive half (largest x-coordinates) of the rink is displayed.
                "defense": The defensive half (smallest x-coordinates) of the rink is displayed.
                "ozone": The offensive zone (blue line to end boards) of the rink is displayed.
                "nzone": The neutral zone (blue line to blue line) of the rink is displayed.
                "dzone": The defensive zone (end boards to blue line) of the rink is displayed.

                If no acceptable values for display_range, xlim, and ylim are provided, will display the full rink.

            xlim: float or (float, float); optional
                If a single float, the lower bound of the x-coordinates to display.  The upper bound will be the
                end of the boards.

                If a tuple, the lower and upper bounds of the x-coordinates to display.

            ylim: float or (float, float); optional
                If a single float, the lower bound of the y-coordinates to display.  The upper bound will be the
                end of the boards.

                If a tuple, the lower and upper bounds of the y-coordinates to display.

            rotation: float; optional
                Degrees to rotate the rink when drawing.

                If used, sets the class attribute.

                0 corresponds to a horizontal rink with largest x and y-coordinates in the top right quadrant.
                90 will rotate so that the largest coordinates are in the top left quadrant.

        Returns:
            matplotlib Axes
        """

        if rotation is not None:
            self._rotation = Affine2D().rotate_deg(rotation)

        if ax is None:
            ax = plt.gca()

        ax.set_aspect("equal")
        ax.axis("off")

        if display_range != "full" or xlim is not None or ylim is not None:
            xlim, ylim = self._get_limits(display_range, xlim, ylim)

        transform = self._get_transform(ax)
        self._boards.draw(ax, transform, xlim, ylim)

        for feature in self._features.values():
            feature_patch = feature.draw(ax, transform, xlim, ylim)

            # Track outer bounds of features not bounded by the boards.
            (feature_xmin, feature_xmax), (feature_ymin, feature_ymax) = feature.get_limits()
            self._feature_xlim = (
                min(self._feature_xlim[0], feature_xmin),
                max(self._feature_xlim[1], feature_xmax)
            )
            self._feature_ylim = (
                min(self._feature_ylim[0], feature_ymin),
                max(self._feature_ylim[1], feature_ymax)
            )

        ax = self.set_display_range(ax, display_range, xlim, ylim)

        return ax

    def set_display_range(self, ax=None, display_range="full", xlim=None, ylim=None):
        """ Set the xlim and ylim for the matplotlib Axes.

        Parameters:
            ax: matplotlib Axes; optional
                Axes in which to set xlim and ylim.

            display_range: {"full", "half", "offense", "defense", "ozone", "dzone"}; default: "full"
                The portion of the rink to display.

                Only affects x-coordinates and can be used in conjunction with ylim, but will be superceded by
                xlim if provided.

                Features not bounded by the boards will only be displayed if display_range is "full".

                "full": The entire length of the rink is displayed.
                "half" or "offense": The offensive half (largest x-coordinates) of the rink is displayed.
                "defense": The defensive half (smallest x-coordinates) of the rink is displayed.
                "ozone": The offensive zone (blue line to end boards) of the rink is displayed.
                "dzone": The defensive zone (end boards to blue line) of the rink is displayed.

                If no acceptable values for display_range, xlim, and ylim are provided, will display the full rink.

            xlim: float or (float, float); optional
                If a single float, the lower bound of the x-coordinates to display.  The upper bound will be the
                end of the boards.

                If a tuple, the lower and upper bounds of the x-coordinates to display.

            ylim: float or (float, float); optional
                If a single float, the lower bound of the y-coordinates to display.  The upper bound will be the
                end of the boards.

                If a tuple, the lower and upper bounds of the y-coordinates to display.
        """

        if ax is None:
            ax = plt.gca()

        if display_range == "full" and xlim is None and ylim is None:
            xlim = self._feature_xlim
            ylim = self._feature_ylim
        else:
            xlim, ylim = self._get_limits(display_range, xlim, ylim)

        # Need each combination of x and y bounds.
        x, y = list(zip(*product(xlim, ylim)))

        # Need to shift coordinates so convert_xy can reverse shift.
        x, y = self.convert_xy(np.array(x) + self.x_shift, np.array(y) + self.y_shift)

        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))

        return ax
