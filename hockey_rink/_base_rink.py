""" Abstract base class for drawing a hockey rink.

Not intended for direct use, only as a parent class.
"""


from abc import ABC
import hockey_rink.rink_feature as rf
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

    def __init__(self):
        self.x_shift = 0
        self.y_shift = 0
        self._rotation = None
        self._feature_xlim = None
        self._feature_ylim = None
        self._features = []
        self._boards_constraint = None
        self._half_nzone_length = 0
        self._blue_line_thickness = 0

    def _initialize_feature(self, params):
        """ Initialize a feature of the rink at each coordinate it's required.

        Parameters:
            params: dict
                The class of the feature and all parameters required to instantiate its class.
        """

        feature_class = params.pop("class")

        xs = np.ravel(params.get("x", [0]))
        ys = np.ravel(params.get("y", [0]))

        x_reflections = [False, True] if params.pop("reflect_x", False) else [False]
        y_reflections = [False, True] if params.pop("reflect_y", False) else [False]

        for x in xs:
            for y in ys:
                for x_reflection in x_reflections:
                    for y_reflection in y_reflections:
                        feature_params = dict(params)

                        feature_params["x"] = x * (-1 if x_reflection else 1)
                        feature_params["y"] = y * (-1 if y_reflection else 1)
                        feature_params["is_reflected_x"] = x_reflection
                        feature_params["is_reflected_y"] = y_reflection

                        self._features.append(feature_class(**feature_params))

    def _add_boards_constraint(self, ax, transform=None):
        """ Add the boards constraint to the rink to avoid features extending beyond boards.

        Parameters:
            ax: matplotlib Axes
                Axes in which to add the constraint.

            transform: matplotlib Transform; optional
                Transform to apply to the constraint.

        Returns:
            matplotlib Polygon
        """

        transform = transform or ax.transData

        constraint = self._boards_constraint.get_polygon()
        constraint.set_transform(transform)
        ax.add_patch(constraint)

        return constraint

    def _get_limits(self, display_range="full", xlim=None, ylim=None, thickness=0):
        """ Return the xlim and ylim values corresponding to the parameters.

        Doesn't allow for coordinates extending beyond the extent of the boards.
        """

        xlim = self.copy_(xlim)
        ylim = self.copy_(ylim)

        # if boards are included in the limits, need to include their thickness
        half_length = self._boards_constraint.length / 2 + thickness

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
                "ozone": (self._half_nzone_length, half_length),
                "nzone": (-self._half_nzone_length - self._blue_line_thickness,
                          self._half_nzone_length + self._blue_line_thickness),
                "dzone": (-half_length, -self._half_nzone_length),
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

        half_width = self._boards_constraint.width / 2 + thickness
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

    def convert_xy(self, x, y):
        """ Convert x,y-coordinates to the scale used for the rink. """

        x = self.copy_(x)
        y = self.copy_(y)

        x = np.ravel(x) - self.x_shift
        y = np.ravel(y) - self.y_shift

        if self._rotation:
            xy = self._rotation.transform(tuple(zip(x, y)))
            return xy[:, 0], xy[:, 1]
        else:
            return x, y

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

        transform = self._get_transform(ax)

        constraint = self._add_boards_constraint(ax, transform)

        for feature in self._features:
            drawn_feature = feature.draw(ax, transform)

            if feature.is_constrained:
                try:
                    drawn_feature.set_clip_path(constraint)
                except AttributeError:
                    pass
            else:
                # need to track outer bounds of unconstrained features to properly set xlim and ylim
                try:
                    visible = feature.visible
                except AttributeError:
                    visible = True

                if visible and not isinstance(feature, rf.Boards):
                    try:
                        feature_x, feature_y = feature.get_polygon_xy()

                        if self._feature_xlim is None:
                            self._feature_xlim = [np.min(feature_x), np.max(feature_x)]
                        else:
                            print(self._feature_xlim)
                            self._feature_xlim = [min(self._feature_xlim[0], np.min(feature_x)),
                                                  max(self._feature_xlim[1], np.max(feature_x))]

                        if self._feature_ylim is None:
                            self._feature_ylim = [np.min(feature_y), np.max(feature_y)]
                        else:
                            self._feature_ylim = [min(self._feature_ylim[0], np.min(feature_y)),
                                                  max(self._feature_ylim[1], np.max(feature_y))]
                    except TypeError:
                        pass

        ax = self.set_display_range(ax, display_range, xlim, ylim)

        return ax

    def set_display_range(self, ax=None, display_range="full", xlim=None, ylim=None):
        """ Set the xlim and ylim for the matplotlib Axes.

        Parameters:
            ax: matplotlib Axes; optional
                Axes in which to set xlim and ylim.

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

        x, y = self._boards_constraint.get_polygon_xy()

        xlim, ylim = self._get_limits(display_range, xlim, ylim,
                                      self._boards_constraint.thickness)

        mask = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
        x = np.concatenate((x[mask], xlim, xlim))
        y = np.concatenate((y[mask], ylim, ylim[::-1]))

        if display_range == "full":
            if self._feature_xlim is not None:
                x = np.concatenate((x, self._feature_xlim))
            if self._feature_ylim is not None:
                y = np.concatenate((y, self._feature_ylim))

        # need to shift so convert_xy can reverse shift
        xs, ys = self.convert_xy(x + self.x_shift, y + self.y_shift)

        ax.set_xlim(np.min(xs), np.max(xs))
        ax.set_ylim(np.min(ys), np.max(ys))

        return ax
