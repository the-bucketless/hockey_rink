""" Module containing Rink, NHLRink, NWHLRink, and IIHFRink for drawing and plotting ice surfaces. """


from hockey_rink._rink_plot import BaseRinkPlot
import hockey_rink.rink_feature as rf
from matplotlib.transforms import Affine2D
import numpy as np


__all__ = ["Rink", "NHLRink", "NWHLRink", "IIHFRink"]


class Rink(BaseRinkPlot):
    """ Rink to draw and plot on with matplotlib.

    Allows for customization to support any number of different possible rink dimensions.

    Default coordinate system and feature dimensions correspond to those used by the NHL.
        ie) All lengths are measured in feet.
            x-coordinates go from -100 to 100 and y-coordinates from -42.5 to 42.5.

    Attributes:
        rotation: matplotlib Affine2D
            Rotation of rink features when drawing.

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

    def __init__(self, rotation=0, x_shift=0, y_shift=0,
                 line_thickness=1 / 6, line_color="red", line_zorder=5,
                 x_dot_to_lines=2, y_dot_to_lines=9 / 12,
                 boards=None, nzone=None, ozone=None, dzone=None,
                 red_line=None, blue_line=None, goal_line=None,
                 trapezoid=None, ref_circle=None, center_dot=None,
                 center_circle=None, faceoff_circle=None, faceoff_dot=None,
                 faceoff_lines=None, crease=None, crease_outline=None,
                 crossbar=None, net=None, **added_features):
        """ Initialize and create the features of the rink.

        Most parameters expect a dict with key/value pairs corresponding to RinkFeature attributes as well as one
        key for indicating the type of RinkFeature class being used.
            eg)
                feature_name = {
                    "class": feature_class,
                    "x": feature_x,
                    "y": feature_y
                    "length": feature_length,
                    "width": feature_width,
                    "thickness": feature_thickness,
                    "radius": feature_radius,
                    "resolution": feature_resolution,
                    "reflect_x": feature_reflect_x,
                    "reflect_y": feature_reflect_y
                    "is_constrained": feature_is_constrained,
                    "visible": feature_visible,
                    ...
                }
        Explanations for the attributes can be found in the RinkFeature documentation.

        The exceptions are:
            Multiple x and y coordinates can be passed as an array_like value.  If multiple values are provided,
            one feature will be created for each combination of coordinates.

            x and y values for faceoff lines correspond to the nearest faceoff dot.  Each coordinate will be
            included in four L shapes (above and right, below and left, etc) with the shape being altered
            accordingly.  The exact coordinate is determined by the values passed to x_dot_to_lines and y_dot_to_lines.

        Other attributes can be provided so long as they can be used by matplotlib's Polygon (such as color or zorder)
        or are appropriate for that particular feature (eg CircularImage requires a path attribute).

        All parameters that expect a dict only require any desired changes to be included in the key/value pairs.
            ie) To update the length of the boards (and, thereby, the rink), all that needs to be passed in is:
                boards={"length": new_length}
        Including other attributes in the dict is unnecessary unless they too require updates.

        Any parameters not included default to NHL dimensions, though they may be affected by changes to other
        parameters.

        To remove a feature from the rink, set visible to False.
            eg) trapezoid={"visible": False}

        New features can be included by passing in a dict with a name not included in the parameter list.
            eg) new_feature: {"class": feature_class, ...}

        The default zorders are:
            1: nzone, ozone, dzone, crease
            5: goal_line, trapezoid, ref_circle, center_circle, faceoff_circle,
                faceoff_dot, faceoff_lines, crease_outline, net
            6: crossbar
            10: red_line, blue_line
            100: boards

        Parameters:
            x_shift: float; default: 0
                Amount x-coordinates are to be shifted.

                When viewing the rink horizontally, the coordinate of the center of the ice surface from left to right.
                    eg) If using data with a coordinate system that goes from 0 to 200, x_shift should be 100.

                The actual coordinates won't be affected.  The purpose is to update the coordinates passed in to
                align with the drawing, not to alter the drawing to align with the coordinates.

            y_shift: float; default: 0
                Amount y-coordinates are to be shifted.

                When viewing the rink horizontally, the coordinate of the center of the ice surface from bottom to top.
                    eg) If using data with a coordinate system that goes from 0 to 85, y_shift should be 42.5.

                The actual coordinates won't be affected.  The purpose is to update the coordinates passed in to
                align with the drawing, not to alter the drawing to align with the coordinates.

            line_thickness: float; default: 1/6
                Thickness of all the thin lines on the ice (eg the goal line and faceoff circles) if not
                otherwise updated.

            line_color: color; default: "red"
                Color of all the thin lines on the ice (eg the goal line and faceoff circles) if not
                otherwise updated.

                An example of how to specify colors can be found at the following link:
                    https://matplotlib.org/stable/tutorials/colors/colors.html#sphx-glr-tutorials-colors-colors-py

            line_zorder: float; default: 5
                The zorder of all thin lines on the ice (eg the goal line and faceoff circles) if not
                otherwise updated.

                Determines which features are drawn first (lower values will cause features to appear under
                other features they may overlap).

            x_dot_to_lines: float; default: 2
                Length-wise distance between a faceoff dot and the L shapes in the faceoff circle.

            y_dot_to_lines = float; default: 9/12
                Width-wise distance between a faceoff dot and the L shapes in the faceoff circle.

            boards: dict; optional
                Attributes to update for the boards.

                Also affects the constraint that prevents features from extending outside the boards.

                The following won't be updated even if values are provided:
                    class
                    x
                    y
                    reflect_x
                    reflect_y
                    is_constrained

            nzone: dict; optional
                Attributes to update for the neutral zone.

                The following won't be updated even if values are provided:
                    class
                    x
                    y
                    width
                    reflect_x
                    reflect_y

            ozone: dict; optional
                Attributes to update for the offensive zone (largest x values).

                The following won't be updated even if values are provided:
                    class
                    x
                    y
                    length
                    reflect_x
                    reflect_y

            dzone: dict; optional
                Attributes to update for the defensive zone (smallest x values).

                The following won't be updated even if values are provided:
                    class
                    x
                    y
                    length
                    reflect_x
                    reflect_y

            red_line: dict; optional
                Attributes to update for the red (center) line.

            blue_line: dict; optional
                Attributes to update for the blue line(s).

            goal_line: dict; optional
                Attributes to update for the goal line(s).

            trapezoid: dict; optional
                Attributes to update for the trapezoid(s) for a goalie's restricted area.

            ref_circle: dict; optional
                Attributes to update for the ref's half circle.

            center_dot: dict; optional
                Attributes to update for the center faceoff dot.

            center_circle: dict; optional
                Attributes to update for the center faceoff circle.

            faceoff_circle: dict; optional
                Attributes to update for the faceoff circle(s) not including the one at
                center ice unless otherwise updated.

                All circles will include hashmarks by default.  To remove hashmarks, change
                the class to RinkCircle (ie faceoff_circle={"class": RinkCircle}).

            faceoff_dot: dict; optional
                Attributes to update for the faceoff dot(s) not including the one at
                center ice unless otherwise updated.

            faceoff_lines: dict; optional
                Attributes to update for the L shapes in the faceoff circles.

            crease: dict; optional
                Attributes to update for the crease(s).

            crease_outline: dict; optional
                Attributes to update for the outline(s) of the crease.

            crossbar: dict; optional
                Attributes to update for the crossbar(s) of the net.

            net: dict; optional
                Attributes to update for the netting of the net.

            added_features: dict; optional
                Any additional features to be added to the rink along with a dict of their
                attributes to be used.
        """

        self._rotation = Affine2D().rotate_deg(rotation)

        self.x_shift = x_shift
        self.y_shift = y_shift

        self._features = []
        self._feature_xlim = None
        self._feature_ylim = None

        boards = boards or {}
        board_params = {
            "length": 200,
            "width": 85,
            "thickness": 1,
            "radius": 28,
            "color": "black",
            "zorder": 100,
        }
        required_boards = {
            "class": rf.Boards,
            "x": 0,
            "y": 0,
            "reflect_x": True,
            "reflect_y": True,
            "is_constrained": False,
        }
        board_params = {**board_params, **boards, **required_boards}
        self._initialize_feature(board_params)

        boards_constraint = {
            "class": rf.BoardsConstraint,
            "length": board_params["length"],
            "width": board_params["width"],
            "radius": board_params["radius"],
            "thickness": board_params["thickness"],
            "reflect_x": False,
            "reflect_y": False,
            "visible": False,
        }
        self._initialize_feature(boards_constraint)
        self._boards_constraint = self._features.pop(-1)

        half_length = board_params["length"] / 2
        half_width = board_params["width"] / 2

        nzone = nzone or {}
        nzone_params = {
            "length": 50,
            "visible": False,
        }
        required_nzone = {
            "class": rf.RinkRectangle,
            "x": 0,
            "y": 0,
            "width": board_params["width"],
            "reflect_x": False,
            "reflect_y": False,
        }
        nzone_params = {**nzone_params, **nzone, **required_nzone}
        self._initialize_feature(nzone_params)

        self._half_nzone_length = nzone_params["length"] / 2

        ozone = ozone or {}
        ozone_length = half_length - self._half_nzone_length
        ozone_params = {
            "visible": False,
        }
        required_ozone = {
            "class": rf.RinkRectangle,
            "x": ozone_length / 2 + self._half_nzone_length,
            "y": 0,
            "length": ozone_length,
            "width": board_params["width"],
            "reflect_x": False,
            "reflect_y": False,
        }
        ozone_params = {**ozone_params, **ozone, **required_ozone}
        self._initialize_feature(ozone_params)

        dzone = dzone or {}
        dzone_params = {
            "visible": False,
        }
        required_dzone = {
            "class": rf.RinkRectangle,
            "x": -ozone_params["x"],
            "y": 0,
            "length": ozone_length,
            "width": board_params["width"],
            "reflect_x": False,
            "reflect_y": False,
        }
        dzone_params = {**dzone_params, **dzone, **required_dzone}
        self._initialize_feature(dzone_params)

        red_line = red_line or {}
        red_line_params = {
            "class": rf.RinkRectangle,
            "length": 1,
            "width": board_params["width"],
            "color": "red",
            "zorder": 10,
        }
        red_line_params.update(red_line)
        self._initialize_feature(red_line_params)

        blue_line = blue_line or {}
        blue_line_params = {
            "class": rf.RinkRectangle,
            "length": 1,
            "width": board_params["width"],
            "reflect_x": True,
            "color": "blue",
            "zorder": 10,
        }
        blue_line_params.update(blue_line)
        blue_line_params["x"] = blue_line_params.get(
            "x", self._half_nzone_length + blue_line_params["length"] / 2)

        self._blue_line_thickness = blue_line_params["length"]
        self._initialize_feature(blue_line_params)

        goal_line = goal_line or {}
        goal_line_params = {
            "class": rf.RinkRectangle,
            "length": line_thickness,
            "width": board_params["width"],
            "reflect_x": True,
            "color": line_color,
            "zorder": line_zorder,
        }
        goal_line_params.update(goal_line)

        # back edge of goal line is 11' from the boards
        goal_line_params["x"] = goal_line_params.get(
            "x", half_length - 11 - goal_line_params["length"] / 2)

        self._initialize_feature(goal_line_params)

        # trapezoid lines go from 11 ft from center to 14 ft from center
        # 11 and 14 refer to the center of the line
        trapezoid = trapezoid or {}
        trapezoid_params = {
            "class": rf.TrapezoidLine,
            "x": goal_line_params["x"] + goal_line_params["length"] / 2,
            "y": 11,
            "width": 3,
            "thickness": line_thickness,
            "reflect_x": True,
            "reflect_y": True,
            "color": line_color,
            "zorder": line_zorder,
        }
        trapezoid_params.update(trapezoid)
        trapezoid_params["length"] = trapezoid_params.get(
            "length", half_length - trapezoid_params["x"])
        self._initialize_feature(trapezoid_params)

        ref_circle = ref_circle or {}
        ref_circle_params = {
            "class": rf.RinkCircle,
            "y": -half_width,
            "thickness": line_thickness,
            "radius": 10,
            "color": line_color,
            "zorder": line_zorder,
        }
        ref_circle_params.update(ref_circle)
        self._initialize_feature(ref_circle_params)

        center_dot = center_dot or {}
        center_dot_params = {
            "class": rf.RinkCircle,
            "radius": red_line_params["length"] / 2,
            "color": "blue",
            "zorder": 11,
        }
        center_dot_params.update(center_dot)
        self._initialize_feature(center_dot_params)

        center_circle = center_circle or {}
        center_circle_params = {
            "class": rf.RinkCircle,
            "thickness": line_thickness,
            "radius": 15,
            "color": "blue",
            "zorder": line_zorder,
        }
        center_circle_params.update(center_circle)
        self._initialize_feature(center_circle_params)

        faceoff_circle = faceoff_circle or {}
        faceoff_circle_params = {
            "class": rf.FaceoffCircle,
            # 20' from front edge of goal line
            "x": goal_line_params["x"] - goal_line_params["length"] / 2 - 20,
            "y": 22,    # 44' between faceoff dots
            "length": 67 / 12,    # 5'7" between inside edges of hashmarks
            "width": 2,    # hashmarks are 2' long
            "thickness": line_thickness,
            "radius": center_circle_params["radius"],
            "resolution": 5000,    # increase resolution to keep lines straight
            "reflect_x": True,
            "reflect_y": True,
            "color": line_color,
            "zorder": line_zorder,
        }
        faceoff_circle_params.update(faceoff_circle)
        self._initialize_feature(faceoff_circle_params)

        ozone_dot_x = np.ravel(faceoff_circle_params["x"])
        dot_y = np.ravel(faceoff_circle_params["y"])

        faceoff_dot = faceoff_dot or {}
        faceoff_dot_params = {
            "class": rf.RinkCircle,
            # ozone faceoff circles and 5' from the blue line
            "x": [*ozone_dot_x, self._half_nzone_length - 5],
            "y": dot_y,
            "length": 16 / 12,    # edge to edge of inner shape
            "thickness": 1 / 12,
            "radius": 1,
            "reflect_x": True,
            "reflect_y": True,
            "color": "red",
            "zorder": 5,
        }

        # split dot into two shapes, one for the outer circle and one for the inner shape
        faceoff_dot_params.update(faceoff_dot)
        inner_dot_params = dict(faceoff_dot_params)
        inner_dot_params["class"] = rf.InnerDot
        self._initialize_feature(faceoff_dot_params)
        self._initialize_feature(inner_dot_params)

        faceoff_lines = faceoff_lines or {}
        faceoff_lines_params = {
            "class": rf.RinkL,
            "x": ozone_dot_x,
            "y": dot_y,
            "length": 4,
            "width": 3,
            "thickness": line_thickness,
            "reflect_x": True,
            "reflect_y": True,
            "color": line_color,
            "zorder": line_zorder,
        }
        faceoff_lines_params.update(faceoff_lines)

        try:
            iter(faceoff_lines_params["x"])
        except TypeError:
            faceoff_lines_params["x"] = [faceoff_lines_params["x"]]

        try:
            iter(faceoff_lines_params["y"])
        except TypeError:
            faceoff_lines_params["y"] = [faceoff_lines_params["y"]]

        # one L for each side of the dot
        for x_side in (1, -1):
            for y_side in (1, -1):
                current_line = dict(faceoff_lines_params)
                current_line["x"] = [x + x_dot_to_lines * x_side for x in current_line["x"]]
                current_line["y"] = [y + y_dot_to_lines * y_side for y in current_line["y"]]

                # change shape by using negative length and/or width
                current_line["length"] = current_line["length"] * x_side
                current_line["width"] = current_line["width"] * y_side

                self._initialize_feature(current_line)

        crease = crease or {}
        crease_params = {
            "class": rf.Crease,
            "x": goal_line_params["x"] - goal_line_params["length"] / 2,
            "length": 4.5,    # 4'6" rectangular section
            "width": 8,    # 8' from outside edge to outside edge
            "radius": 1.5,    # total length 6'
            "reflect_x": True,
            "reflect_y": True,
            "color": "lightblue",
        }
        crease_params.update(crease)

        crease_outline = crease_outline or {}
        crease_outline_params = dict(crease_params)
        crease_outline_params["thickness"] = line_thickness
        crease_outline_params["color"] = line_color
        crease_outline_params["zorder"] = line_zorder
        crease_outline_params.update(crease_outline)

        self._initialize_feature(crease_params)
        self._initialize_feature(crease_outline_params)

        crossbar = crossbar or {}
        crossbar_params = {
            "class": rf.Crossbar,
            "x": goal_line_params["x"] - goal_line_params["length"] / 2,
            # posts are 2 3/8" wide => half = 19/16"
            "radius": 19 / 16 / 12,
            "reflect_x": True,
            "color": "red",
            "zorder": 6,
        }
        crossbar_params.update(crossbar)
        crossbar_params["width"] = crossbar_params.get(
            "width", 6 + crossbar_params["radius"])
        self._initialize_feature(crossbar_params)

        net = net or {}
        net_params = {
            "class": rf.Net,
            "x": crossbar_params["x"] + crossbar_params["radius"] * 2,
            "length": 40 / 12,    # 40" deep
            "width": crossbar_params["width"] + crossbar_params["radius"],
            "thickness": 88 / 12,    # max width
            "radius": 20 / 12,
            "reflect_x": True,
            "color": "grey",
            "zorder": 5,
        }
        net_params.update(net)
        self._initialize_feature(net_params)

        for added_feature in added_features.values():
            self._initialize_feature(added_feature)


class NHLRink(Rink):
    """ Version of Rink class based off a typical NHL ice surface.

    Includes an additional feature "crease_notch" for the little notches inside the crease.

    See Rink for full documentation.
    """

    def __init__(self, **kwargs):
        crease = kwargs.get("crease", {})
        line_thickness = kwargs.get("line_thickness", 1 / 6)
        half_length = kwargs.get("boards", {}).get("length", 200) / 2
        half_goal_line_thickness = kwargs.get("goal_line", {}).get("length", line_thickness) / 2
        goal_line_x = kwargs.get("goal_line", {}).get(
            "x", half_length - 11 - half_goal_line_thickness)
        crease_thickness = crease.get("thickness", line_thickness)
        notch_width = 5 / 12

        nhl_updates = {
            "crease_notch": {
                "class": rf.RinkRectangle,
                "x": goal_line_x - 4 - crease_thickness / 2,
                "y": ((crease.get("width", 8) - notch_width) / 2
                      - crease_thickness),
                "length": crease_thickness,
                "width": notch_width,
                "reflect_x": crease.get("reflect_x", True),
                "reflect_y": crease.get("reflect_y", True),
                "color": kwargs.get("line_color", "red"),
                "zorder": kwargs.get("line_zorder", 5),
                "visible": crease.get("visible", True),
            },
        }

        kwargs["crease_notch"] = {**nhl_updates["crease_notch"],
                                  **kwargs.get("crease_notch", {})}

        super().__init__(**kwargs)


class NWHLRink(NHLRink):
    """ Version of Rink class based off of the NWHL rink in the 2021 playoffs.

    Includes additional features of "logo" for the logo at center ice and
    "crease_notch" for the little notches inside the crease.

    See Rink for full documentation.
    """

    def __init__(self, **kwargs):
        half_width = kwargs.get("boards", {}).get("width", 85) / 2
        center_radius = kwargs.get("center_circle", {}).get("radius", 15)

        nwhl_updates = {
            "nzone": {"color": "#B266FF", "visible": True},
            "ref_circle": {"y": half_width},
            "center_circle": {"thickness": 2, "color": "#003366", "zorder": 12},
            "center_dot": {"visible": False},
            "logo": {"class": rf.CircularImage,
                     "path": "https://raw.githubusercontent.com/the-bucketless/hockey_rink/master/images/nwhl_logo.png",
                     "radius": center_radius, "is_constrained": False, "zorder": 11}
        }

        for k, v in nwhl_updates.items():
            kwargs[k] = {**v, **kwargs.get(k, {})}

        super().__init__(**kwargs)


class IIHFRink(Rink):
    """ Version of Rink class with dimensions based off of IIHF regulations.

    Includes an additional feature "crease_notch" for the little notches inside the crease.

    See Rink for full documentation.
    """

    def __init__(self, **kwargs):
        iihf_updates = {
            "boards": {"length": 197, "width": 98.4},
            "nzone": {"length": 47},
            "trapezoid": {"visible": False},
            "crease": {"length": 0, "width": 11.6, "radius": 5.9},
        }

        line_thickness = kwargs.get("line_thickness", 1 / 6)
        half_goal_line_thickness = kwargs.get("goal_line", {}).get("length", line_thickness) / 2
        boards_length = kwargs.get("boards", {}).get("length", 197)
        iihf_updates["goal_line"] = {"x": boards_length / 2 - 13.1 + half_goal_line_thickness}
        goal_line_x = (kwargs.get("goal_line", iihf_updates["goal_line"])["x"]
                       - half_goal_line_thickness)

        iihf_updates["faceoff_circle"] = {"x": goal_line_x - 22}

        crease = kwargs.get("crease", {})
        crease_thickness = crease.get("thickness", line_thickness)
        notch_size = 5 / 12

        iihf_updates["crease_notch"] = {
            "class": rf.RinkL,
            "x": goal_line_x - 4,
            "y": 4,
            "length": notch_size,
            "width": -notch_size,
            "thickness": crease_thickness,
            "reflect_x": crease.get("reflect_x", True),
            "reflect_y": crease.get("reflect_y", True),
            "color": kwargs.get("line_color", "red"),
            "zorder": kwargs.get("line_zorder", 5),
            "visible": crease.get("visible", True),
        }

        for k, v in iihf_updates.items():
            kwargs[k] = {**v, **kwargs.get(k, {})}

        super().__init__(**kwargs)
