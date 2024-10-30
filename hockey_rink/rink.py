""" Module containing Rink, NHLRink, NWHLRink, and IIHFRink for drawing and plotting ice surfaces. """


from hockey_rink._rink_plot import BaseRinkPlot
from hockey_rink.rink_feature import *
from itertools import product
import numpy as np
from pathlib import Path


__all__ = ["BlankRink", "Rink", "NHLRink", "NWHLRink", "IIHFRink", "OldIIHFRink"]

CURRENT_DIR = Path(__file__).resolve().parent


class BlankRink(BaseRinkPlot):
    """ Rink to draw and plot on with matplotlib.

    Allows for customization to support any number of different possible rink dimensions.

    Default coordinate system and feature dimensions correspond to those used by the NHL. ie) All lengths are measured
    in feet and coordinates range from eg) -100 to 100 in x-coordinates for an NHL rink.

    By default, will include the following features:
            nzone
            ozone
            dzone
            red_line
            blue_line
            goal_line
            trapezoid
            ref_circle
            center_circle
            center_dot
            faceoff_circle
            faceoff_dot
            faceoff_lines
            crease
            crease_outline
            crossbar
            net
            ice (by default, not visible)
            crease_notch (in all but Rink)
            logo (only in NWHLRink)

        The ice is the only feature that, by default, isn't visible. When visible, it draws an image of ice on the rink.

    All default features include linewidth to ensure they appear. This also may cause them to be larger relative other
    features than they ought to be. If using larger figure sizes, linewidth on all features can be reduced or set to 0.

    Attributes:
        rotation: float
            Degree the rink will be rotated. This can be altered for different Axes, but rotation will remain the
            default when drawing.

        x_shift: float
            Amount x-coordinates are to be shifted.

            When viewing the rink horizontally, the coordinate of the center of the ice surface from left to right.
                eg) If using data with a coordinate system that goes from 0 to 200, x_shift should be 100.

            The actual coordinates won't be affected. The purpose is to update the coordinates passed in to
            align with the drawing, not to alter the drawing to align with the coordinates.

        y_shift: float
            Amount y-coordinates are to be shifted.

            When viewing the rink horizontally, the coordinate of the center of the ice surface from bottom to top.
                eg) If using data with a coordinate system that goes from 0 to 85, y_shift should be 42.5.

            The actual coordinates won't be affected. The purpose is to update the coordinates passed in to
            align with the drawing, not to alter the drawing to align with the coordinates.
    """

    def __init__(
            self,
            rotation=0, x_shift=0, y_shift=0, alpha=None, linewidth=None,
            line_thickness=1 / 6, line_color="red", line_zorder=5,
            x_dot_to_lines=2, y_dot_to_lines=9 / 12, goal_line_to_dot=20,
            boards=None,
            **features,
    ):
        """ Initialize and create the features of the rink.

        The features parameters allows for both updating default features and creating new features.

        The defaults features are:
            nzone
            ozone
            dzone
            red_line
            blue_line
            goal_line
            trapezoid
            ref_circle
            center_circle
            center_dot
            faceoff_circle
            faceoff_dot
            faceoff_lines
            crease
            crease_outline
            crossbar
            net
            ice (by default, not visible)
            crease_notch (in all but Rink)
            logo (only in NWHLRink)

        The ice is the only feature that, by default, isn't visible. When visible, it draws an image of ice on the rink.

        Updates to existing features and new features both expect a dict with key/value pairs corresponding to
        RinkFeature attributes. Additionally, features expect a key for the feature_class with a value indicating the
        type of RinkFeature class being used.

            eg)
                feature_name = {
                    "feature_class": feature_class,
                    "x": feature_x,
                    "y": feature_y
                    "length": feature_length,
                    "width": feature_width,
                    "thickness": feature_thickness,
                    "radius": feature_radius,
                    "resolution": feature_resolution,
                    "is_reflected_x": feature_is_reflected_x,
                    "is_reflected_y": feature_is_reflected_y
                    "visible": feature_visible,
                    "rotation": feature_rotation,
                    "clip_xy": feature_clip_xy,
                    ...
                }
        Explanations for the attributes can be found in the RinkFeature documentation.

        Some exceptions are:
            Multiple x and y coordinates can be passed as an array-like value. If multiple values are provided,
            one feature will be created for each combination of coordinates. Likewise, for multiple is_reflected_x
            and is_reflected_y values.

            x and y values for faceoff lines correspond to the nearest faceoff dot. Each coordinate will be
            included in four L shapes (above and right, below and left, etc) with the shape being altered accordingly.
            The exact coordinate is determined by the values passed to x_dot_to_lines and y_dot_to_lines.

            clip_xy also accepts a boolean value and defaults to True for all features.
                When True, the boards will be used as the clip path.
                When False, no clip path is used.

        Other attributes can be provided so long as they can be used by matplotlib's Polygon (such as color or zorder)
        or are appropriate for that particular feature (eg RinkImage accepts an image_path attribute).

        All parameters that expect a dict only require any desired changes to be included in the key/value pairs.
            ie) To update the length of the boards (and, thereby, the rink), all that needs to be passed in is:
                boards={"length": new_length}
        Including other attributes in the dict is unnecessary unless they too require updates.

        Any parameters not included will be supplied with defaults based on this rink type, though they may be affected
        by changes to other parameters.

        To remove a feature from the rink, set visible to False.
            eg) trapezoid={"visible": False}

        New features can be included by passing in a dict with a name not included in the default list.
            eg) new_feature={"feature_class": feature_class, ...}

        The default zorders are:
            1: nzone, ozone, dzone
            1.5: ice
            2: crease
            5: goal_line, trapezoid, ref_circle, center_circle, faceoff_circle,
                faceoff_dot, faceoff_lines, crease_outline, net
            6: crossbar
            10: red_line, blue_line
            100: boards

        Parameters:
            rotation: float (default=0)
                Degree to rotate the rink.

            x_shift: float (default=0)
                Amount x-coordinates are to be shifted.

                When viewing the rink horizontally, the coordinate of the center of the ice surface from left to right.
                    eg) If using data with a coordinate system that goes from 0 to 200, x_shift should be 100.

                The actual coordinates won't be affected. The purpose is to update the coordinates passed in to
                align with the drawing, not to alter the drawing to align with the coordinates.

            y_shift: float (default=0)
                Amount y-coordinates are to be shifted.

                When viewing the rink horizontally, the coordinate of the center of the ice surface from bottom to top.
                    eg) If using data with a coordinate system that goes from 0 to 85, y_shift should be 42.5.

                The actual coordinates won't be affected. The purpose is to update the coordinates passed in to
                align with the drawing, not to alter the drawing to align with the coordinates.

            alpha: float (optional)
                The alpha blending value, between 0 (transparent) and 1 (opaque).

                If not None, will be used for all features of the rink that don't override it.

            linewidth: float (optional)
                The linewidth to use in creating the Polygons for features. By default, all features include linewidth
                to ensure they appear. This may also cause them to be slightly larger than they should be relative other
                features. When using a larger figsize, linewidth can be reduced or set to 0.

            line_thickness: float (default=1/6)
                Thickness of all the thin lines on the ice (eg the goal line and faceoff circles) if not
                otherwise updated.

                Ignored by BlankRink.

            line_color: color (default="red")
                Color of all the thin lines on the ice (eg the goal line and faceoff circles) if not
                otherwise updated.

                Ignored by BlankRink.

                An example of how to specify colors can be found at the following link:
                    https://matplotlib.org/stable/tutorials/colors/colors.html#sphx-glr-tutorials-colors-colors-py

            line_zorder: float (default=5)
                The zorder of all thin lines on the ice (eg the goal line and faceoff circles) if not
                otherwise updated.

                Determines which features are drawn first (lower values will cause features to appear under
                other features they may overlap).

                Ignored by BlankRink.

            x_dot_to_lines: float (default=2)
                Length-wise distance between a faceoff dot and the L shapes in the faceoff circle.

                Ignored by BlankRink.

            y_dot_to_lines: float (default=9/12)
                Width-wise distance between a faceoff dot and the L shapes in the faceoff circle.

                Ignored by BlankRink.

            goal_line_to_dot: float (default=20 except for IIHFRink which is 22)
                Distance between the goal line and the faceoff dots.

                Ignored by BlankRink.

            boards: dict (optional)
                Attributes to update for the boards.

                Also affects the constraint that prevents features from extending outside the boards.

            features: dict (optional)
                Updates to default features and new features to be added, as described above.
        """

        super().__init__(rotation, x_shift, y_shift, alpha, linewidth, boards)

        features = self._compute_feature_params(
            features,
            line_thickness, line_color, line_zorder,
            x_dot_to_lines, y_dot_to_lines, goal_line_to_dot,
        )

        for feature_name, feature_params in features.items():
            self._initialize_feature(feature_name, feature_params, alpha, linewidth)

    def _compute_feature_params(
            self,
            features,
            line_thickness, line_color, line_zorder,
            x_dot_to_lines, y_dot_to_lines, goal_line_to_dot,
    ):
        features = features or {}

        feature_defaults = {
            "ice": {
                "feature_class": RinkImage,
                "visible": False,
                "zorder": 1.5,
                "image_path": CURRENT_DIR.parent / "images" / "ice.png",
            },
        }

        # Update any missing values with defaults.
        features = self._merge_params(features, feature_defaults)

        features["ice"]["length"] = features["ice"].get("length", self._boards.length)
        features["ice"]["width"] = features["ice"].get("width", self._boards.width)

        return features

    @staticmethod
    def _merge_params(features, feature_defaults):
        """ Update missing values in features with defaults. """
        feature_names = set(features.keys()).union(feature_defaults.keys())
        return {
            feature_name: {
                **feature_defaults.get(feature_name, {}),
                **features.get(feature_name, {})
            }
            for feature_name in feature_names
        }


class Rink(BlankRink):
    def _compute_feature_params(
            self,
            features,
            line_thickness, line_color, line_zorder,
            x_dot_to_lines, y_dot_to_lines, goal_line_to_dot,
    ):
        """ Update any missing parameters for features using defaults for this rink. """

        features = super()._compute_feature_params(
            features,
            line_thickness, line_color, line_zorder,
            x_dot_to_lines, y_dot_to_lines, goal_line_to_dot
        )

        half_length = self._boards.length / 2
        half_width = self._boards.width / 2

        feature_defaults = {
            "nzone": {
                "feature_class": RinkRectangle,
                "length": 50,
                "width": self._boards.width,
                "color": "white",
            },
            "ozone": {
                "feature_class": RinkRectangle,
                "width": self._boards.width,
                "color": "white",
            },
            "dzone": {
                "feature_class": RinkRectangle,
                "width": self._boards.width,
                "color": "white",
            },
            "red_line": {
                "feature_class": RinkRectangle,
                "length": 1,
                "width": self._boards.width,
                "color": "red",
                "zorder": 10,
            },
            "blue_line": {
                "feature_class": RinkRectangle,
                "length": 1,
                "width": self._boards.width,
                "is_reflected_x": [False, True],
                "color": "blue",
                "zorder": 10,
            },
            "goal_line": {
                "feature_class": RinkRectangle,
                "length": line_thickness,
                "width": self._boards.width,
                "is_reflected_x": [False, True],
                "color": line_color,
                "zorder": line_zorder,
            },
            "trapezoid": {
                "feature_class": TrapezoidLine,
                "y": 11,
                "width": 3,  # 11' from center to 14' from center.
                "thickness": line_thickness,
                "is_reflected_x": [False, True],
                "is_reflected_y": [False, True],
                "color": line_color,
                "zorder": line_zorder,
            },
            "ref_circle": {
                "feature_class": RinkCircle,
                "y": -half_width,
                "thickness": line_thickness,
                "radius": 10,
                "color": line_color,
                "zorder": line_zorder,
            },
            "center_dot": {
                "feature_class": RinkCircle,
                "color": "blue",
                "zorder": 11,
            },
            "center_circle": {
                "feature_class": RinkCircle,
                "thickness": line_thickness,
                "radius": 15,
                "color": "blue",
                "zorder": line_zorder,
            },
            "faceoff_circle": {
                "feature_class": FaceoffCircle,
                "y": 22,  # 44' between faceoff dots.
                "length": 67 / 12,  # 5'7" between inside edges of hashmarks.
                "width": 2,  # Hashmarks are 2' long.
                "thickness": line_thickness,
                "resolution": 5000,  # Increase resolution to keep lines straight.
                "is_reflected_x": [False, True],
                "is_reflected_y": [False, True],
                "color": line_color,
                "zorder": line_zorder,
            },
            "faceoff_dot": {
                "feature_class": FaceoffDot,
                "length": 16 / 12,  # Edge of circle to edge of inner shape.
                "thickness": 1 / 12,
                "radius": 1,
                "is_reflected_x": [False, True],
                "is_reflected_y": [False, True],
                "color": "red",
                "zorder": 5,
            },
            "faceoff_lines": {
                "feature_class": RinkL,
                "length": 4,
                "width": 3,
                "thickness": line_thickness,
                "is_reflected_x": [False, True],
                "is_reflected_y": [False, True],
                "color": line_color,
                "zorder": line_zorder,
            },
            "crease": {
                "feature_class": Crease,
                "length": 4.5,  # 4'6" rectangular section.
                "width": 8,  # 8' from outside edge to outside edge.
                "radius": 1.5,  # 6' total length.
                "is_reflected_x": [False, True],
                "color": "lightblue",
                "zorder": 2,
            },
            "crease_outline": {
                "thickness": line_thickness,
                "color": line_color,
                "zorder": line_zorder,
            },
            "crossbar": {
                "feature_class": Crossbar,
                "radius": 19 / 16 / 12,  # Posts are 2+3/8" wide, half = 19/16"
                "is_reflected_x": [False, True],
                "resolution": 10,
                "color": "red",
                "zorder": 6,
            },
            "net": {
                "feature_class": Net,
                "length": 40 / 12,  # 40' deep.
                "thickness": 88 / 12,  # Width from outer edge to outer edge.
                "radius": 20 / 12,
                "is_reflected_x": [False, True],
                "color": "grey",
                "zorder": 5,
            },
        }

        # Update any missing values with defaults.
        features = self._merge_params(features, feature_defaults)

        # Update for defaults that depend on other features.
        half_nzone_length = features["nzone"]["length"] / 2
        ozone_length = half_length - half_nzone_length

        features["ozone"]["x"] = features["ozone"].get("x", ozone_length / 2 + half_nzone_length)
        features["ozone"]["length"] = features["ozone"].get("length", ozone_length)

        features["dzone"]["x"] = features["dzone"].get("x", -features["ozone"]["x"])
        features["dzone"]["length"] = features["dzone"].get("length", ozone_length)

        features["blue_line"]["x"] = features["blue_line"].get(
            "x",
            half_nzone_length + features["blue_line"]["length"] / 2
        )

        # Back edge of goal line is 11' from the boards.
        features["goal_line"]["x"] = features["goal_line"].get(
            "x",
            half_length - 11 - features["goal_line"]["length"] / 2
        )

        features["trapezoid"]["x"] = features["trapezoid"].get(
            "x",
            features["goal_line"]["x"] + features["goal_line"]["length"] / 2
        )
        features["trapezoid"]["length"] = features["trapezoid"].get("length", half_length - features["trapezoid"]["x"])

        features["center_dot"]["radius"] = features["center_dot"].get("radius", features["red_line"]["length"] / 2)

        features["faceoff_circle"]["x"] = features["faceoff_circle"].get(
            "x",
            features["goal_line"]["x"] - features["goal_line"]["length"] / 2 - goal_line_to_dot
        )
        features["faceoff_circle"]["radius"] = features["faceoff_circle"].get(
            "radius",
            features["center_circle"]["radius"]
        )

        # Ozone dots in center of circle, nzone dots 5' from the blue line.
        ozone_dot_x = np.ravel(features["faceoff_circle"]["x"])
        dot_y = np.ravel(features["faceoff_circle"]["y"])
        features["faceoff_dot"]["x"] = features["faceoff_dot"].get("x", [*ozone_dot_x, half_nzone_length - 5])
        features["faceoff_dot"]["y"] = features["faceoff_dot"].get("y", dot_y)

        features["faceoff_lines"]["x"] = features["faceoff_lines"].get("x", ozone_dot_x)
        features["faceoff_lines"]["y"] = features["faceoff_lines"].get("y", dot_y)
        faceoff_lines = features.pop("faceoff_lines")

        # One L for each side of the dot.
        for i, (x_side, y_side) in enumerate(product((1, -1), (1, -1))):
            x_map = {1: "right", -1: "left"}
            y_map = {1: "top", -1: "bottom"}

            line_params = dict(faceoff_lines)
            line_params["x"] = [x + x_dot_to_lines * x_side for x in line_params["x"]]
            line_params["y"] = [y + y_dot_to_lines * y_side for y in line_params["y"]]

            # Change shape by using negative length and/or width.
            line_params["length"] *= x_side
            line_params["width"] *= y_side

            features[f"{y_map[y_side]}_{x_map[x_side]}_faceoff_line"] = line_params

        back_goal_line_x = features["goal_line"]["x"] - features["goal_line"]["length"] / 2
        features["crease"]["x"] = features["crease"].get("x", back_goal_line_x)

        # Update crease outline based on crease.
        for k, v in features["crease"].items():
            if k not in features["crease_outline"]:
                features["crease_outline"][k] = v

        features["crossbar"]["x"] = features["crossbar"].get(
            "x",
            back_goal_line_x
        )
        features["crossbar"]["width"] = features["crossbar"].get("width", 6 + features["crossbar"]["radius"])
        features["crossbar"]["visible"] = features["crossbar"].get("visible", features["net"].get("visible", True))

        features["net"]["x"] = features["net"].get("x", features["crossbar"]["x"] + features["crossbar"]["radius"] * 2)
        features["net"]["width"] = features["net"].get(
            "width",
            features["crossbar"]["width"] + features["crossbar"]["radius"]
        )

        return features


class NHLRink(Rink):
    def _compute_feature_params(
        self,
        features,
        line_thickness, line_color, line_zorder,
        x_dot_to_lines, y_dot_to_lines, goal_line_to_dot
    ):
        features = super()._compute_feature_params(
            features,
            line_thickness, line_color, line_zorder,
            x_dot_to_lines, y_dot_to_lines, goal_line_to_dot
        )

        notch_width = 5 / 12
        crease_thickness = features["crease_outline"]["thickness"]

        crease_notch = {
            "feature_class": RinkRectangle,
            "x": features["goal_line"]["x"] - 4 - crease_thickness / 2,
            "y": (features["crease"]["width"] - notch_width) / 2 - crease_thickness,
            "length": crease_thickness,
            "width": notch_width,
            "is_reflected_x": features["crease"]["is_reflected_x"],
            "is_reflected_y": [False, True],
            "color": line_color,
            "zorder": line_zorder,
            "visible": features["crease_outline"].get("visible", True),
        }

        features["crease_notch"] = {**crease_notch, **features.get("crease_notch", {})}

        return features


class NWHLRink(NHLRink):
    def _compute_feature_params(
        self,
        features,
        line_thickness, line_color, line_zorder,
        x_dot_to_lines, y_dot_to_lines, goal_line_to_dot
    ):
        half_width = features.get("boards", {}).get("width", 85) / 2
        center_radius = features.get("center_circle", {}).get("radius", 15)
        center_thickness = 2

        feature_defaults = {
            "nzone": {
                "length": 60,
                "color": "#B266FF",
            },
            "ref_circle": {"y": half_width},
            "center_circle": {
                "thickness": 2,
                "color": "#003366",
                "zorder": 12,
                "linewidth": 0,  # Avoid drawing line in circle when alpha isn't 1.
            },
            "center_dot": {"visible": False},
            "trapezoid": {"visible": False},
            "logo": {
                "feature_class": CircularImage,
                "thickness": center_thickness,
                "radius": center_radius - center_thickness,
                "zorder": 11,
                "image_path": CURRENT_DIR.parent / "images" / "nwhl_logo.png",
            },
            "red_line": {
                "feature_class": LowerInwardArcRectangle,
                "radius": center_radius,
                "is_reflected_y": [False, True],
            },
        }

        features = self._merge_params(features, feature_defaults)

        features = super()._compute_feature_params(
            features,
            line_thickness, line_color, line_zorder,
            x_dot_to_lines, y_dot_to_lines, goal_line_to_dot
        )

        return features


class OldIIHFRink(Rink):
    def __init__(
        self,
        rotation=0, x_shift=0, y_shift=0, alpha=None, linewidth=None,
        line_thickness=1 / 6, line_color="red", line_zorder=5,
        x_dot_to_lines=2, y_dot_to_lines=9 / 12, goal_line_to_dot=22,
        boards=None,
        **features,
    ):
        boards = boards or {}
        boards["length"] = boards.get("length", 197)
        boards["width"] = boards.get("width", 98.4)

        super().__init__(
            rotation, x_shift, y_shift, alpha, linewidth,
            line_thickness, line_color, line_zorder,
            x_dot_to_lines, y_dot_to_lines, goal_line_to_dot,
            boards,
            **features,
        )

    def _compute_feature_params(
        self,
        features,
        line_thickness, line_color, line_zorder,
        x_dot_to_lines, y_dot_to_lines, goal_line_to_dot
    ):
        half_goal_line_thickness = features.get("goal_line", {}).get("length", line_thickness) / 2

        feature_defaults = {
            "nzone": {"length": 47},
            "trapezoid": {"visible": False},
            "crease": {"length": 0, "width": 11.6, "radius": 5.9},
            "goal_line": {"x": self._boards.length / 2 - 13.1 + half_goal_line_thickness},
        }

        features = self._merge_params(features, feature_defaults)

        features = super()._compute_feature_params(
            features,
            line_thickness, line_color, line_zorder,
            x_dot_to_lines, y_dot_to_lines, goal_line_to_dot
        )

        notch_size = 5 / 12
        crease_thickness = features["crease_outline"]["thickness"]

        crease_notch = {
            "feature_class": RinkL,
            "x": features["goal_line"]["x"] - half_goal_line_thickness - 4,
            "y": 4,
            "length": notch_size,
            "width": -notch_size,
            "thickness": crease_thickness,
            "is_reflected_x": features["crease"]["is_reflected_x"],
            "is_reflected_y": [False, True],
            "color": line_color,
            "zorder": line_zorder,
            "visible": features["crease_outline"].get("visible", True),
        }

        features["crease_notch"] = {**crease_notch, **features.get("crease_notch", {})}

        return features


class IIHFRink(NHLRink):
    def __init__(
        self,
        rotation=0, x_shift=0, y_shift=0, alpha=None, linewidth=None,
        line_thickness=1 / 6, line_color="red", line_zorder=5,
        x_dot_to_lines=2, y_dot_to_lines=9 / 12, goal_line_to_dot=22,
        boards=None,
        **features,
    ):
        boards = boards or {}
        boards["length"] = boards.get("length", 197)
        boards["width"] = boards.get("width", 98.4)

        super().__init__(
            rotation, x_shift, y_shift, alpha, linewidth,
            line_thickness, line_color, line_zorder,
            x_dot_to_lines, y_dot_to_lines, goal_line_to_dot,
            boards,
            **features,
        )

    def _compute_feature_params(
        self,
        features,
        line_thickness, line_color, line_zorder,
        x_dot_to_lines, y_dot_to_lines, goal_line_to_dot
    ):
        half_goal_line_thickness = features.get("goal_line", {}).get("length", line_thickness) / 2

        feature_defaults = {
            "nzone": {"length": 49.2},
            "goal_line": {"x": self._boards.length / 2 - 13.1 + half_goal_line_thickness},
        }

        features = self._merge_params(features, feature_defaults)

        features = super()._compute_feature_params(
            features,
            line_thickness, line_color, line_zorder,
            x_dot_to_lines, y_dot_to_lines, goal_line_to_dot
        )

        return features
