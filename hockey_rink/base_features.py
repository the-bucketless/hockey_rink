""" Module containing all provided rink features as well as the abstract base class.

Currently available features are:
    RinkRectangle
    RinkCircle
    TrapezoidLine
    InnerDot
    FaceoffCircle
    RinkL
    Crease
    Crossbar
    Net
    CircularImage
    LowerInwardArcRectangle
    RoundedRectangle
"""


__all__ = [
    "RinkFeature",
    "RinkRectangle",
    "RinkCircle",
    "TrapezoidLine",
    "InnerDot",
    "FaceoffCircle",
    "RinkL",
    "Crease",
    "Crossbar",
    "Net",
    "CircularImage",
    "LowerInwardArcRectangle",
    "RoundedRectangle",
]


from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np


class RinkFeature(ABC):
    """ Abstract base class for features of a rink to draw.

    Child classes should override the get_centered_xy method which determines the coordinates of the plt.Polygon
    representing the feature, were it to be recorded at center. This will be called by the get_polygon_xy method to
    shift all coordinates to the correct place.

    Attributes:
        x: float
            Typically, the center x-coordinate of the feature.

        y: float
            Typically, the center y-coordinate of the feature.

        length: float
            Typically, the size of the feature from left to right.

        width: float
            Typically, the size of the feature from bottom to top.

        thickness: float

        radius: float

        resolution: int
            The number of coordinates used in creating arcs.

        is_reflected_x: bool
            Whether or not the x-coordinates are to be reflected.

        is_reflected_y: bool
            Whether or not the y-coordinates are to be reflected.

        visible: bool
            Whether or not the feature will be drawn.

        clip_xy: (np.array, np.array)
            Coordinates used to clip the feature's coordinates when drawing.

        polygon_kwargs: dict
            Any additional arguments to be passed to plt.Polygon.
    """

    def __init__(
        self,
        x=0, y=0,
        length=0, width=0, thickness=0,
        radius=0, resolution=500,
        is_reflected_x=False, is_reflected_y=False,
        visible=True, color=None, zorder=None,
        clip_xy=None,
        **polygon_kwargs,
    ):
        """ Initialize attributes.

        Parameters:
            x: float (default=0)
            y: float (default=0)
            length: float (default=0)
            width: float (default=0)
            thickness: float (default=0)
            radius: float (default=0)
            resolution: int (default=500)
            is_reflected_x: bool (default=False)
            is_reflected_y: bool (default=False)
            visible: bool (default=True)
            color: color (optional)
            zorder: float (optional)
            clip_xy: (np.array, np.array) (optional)
            polygon_kwargs: dict (optional)
        """

        self.x = x
        self.y = y
        self.length = length
        self.width = width
        self.thickness = thickness
        self.radius = radius
        self.resolution = resolution
        self.is_reflected_x = is_reflected_x
        self.is_reflected_y = is_reflected_y
        self.visible = visible
        self.clip_xy = clip_xy
        self.polygon_kwargs = polygon_kwargs

        if color is not None:
            self.polygon_kwargs["color"] = color
        self.polygon_kwargs["zorder"] = zorder

    @abstractmethod
    def get_centered_xy(self):
        """ Determines the x and y-coordinates necessary to create Polygon of the feature were it to be placed
        at (0, 0).

        Returns:
            np.array, np.array
        """

        pass

    @staticmethod
    def arc_coords(center, width, height=None, thickness=0, theta1=0, theta2=360, resolution=500):
        """ Generates the x and y-coordinates for an arc.

        When thickness is not 0, will include an inside and an outside arc. The outside arc will be in the reverse
        direction of the inside arc. All coordinates will be concatenated together.

        Adapted from:
        https://stackoverflow.com/questions/30642391/how-to-draw-a-filled-arc-in-matplotlib

        Parameters:
            center: tuple

            width: float

            height: float (optional)
                If None, height will be equal to width.

            thickness: float (default=0)
                If not 0, two arcs will be created and concatenated together: one with the width and height provided
                and one with thickness added to both width and height. The outside arc will be in the reverse direction
                of the inside arc.

            theta1: float (default=0)
                Degree at which to start the arc.
                    0: Right
                    90: Above
                    180: Left
                    270: Below

            theta2: float (default=360)
                Degree at which to end the arc.
                    0: Right
                    90: Above
                    180: Left
                    270: Below

            resolution: int (default=500)
                The number of coordinates used in creating the arc. Using larger numbers results in slower drawing
                but finer detail. The default should be more than enough in most cases, but it may need scaling for
                larger images.

        Returns:
            np.array, np.array
        """

        height = width if height is None else height

        theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
        x = width * np.cos(theta) + center[0]
        y = height * np.sin(theta) + center[1]

        if thickness > 0:
            # Reverse the direction for the outside arc.
            theta = np.linspace(np.radians(theta2), np.radians(theta1), resolution)
            x = np.concatenate((x, (width + thickness) * np.cos(theta) + center[0]))
            y = np.concatenate((y, (height + thickness) * np.sin(theta) + center[1]))

        return x, y

    @staticmethod
    def tangent_point(center, radius, point):
        """ Finds the x and y-coordinates on a circle that are tangent to a point outside the circle.

        Adapted from:
        https://stackoverflow.com/a/49987361

        Parameters:
            center: tuple
            radius: float
            point: tuple

        Returns:
            float, float
        """

        dx = point[0] - center[0]
        dy = point[1] - center[1]
        center_to_point = (dx ** 2 + dy ** 2) ** 0.5
        theta = np.arccos(radius / center_to_point)
        angle = np.arctan2(dy, dx) - theta * np.sign(point[1])

        return (
            center[0] + radius * np.cos(angle),
            center[1] + radius * np.sin(angle)
        )

    def get_polygon(self):
        """ Creates the plt.Polygon representing the feature. """

        polygon_x, polygon_y = self.get_polygon_xy()

        return plt.Polygon(
            tuple(zip(polygon_x, polygon_y)),
            **self.polygon_kwargs,
        )

    def get_polygon_xy(self):
        """ Determines the x and y-coordinates necessary for creating the plt.Polygon representing the feature. """

        x, y = self.get_centered_xy()

        if self.is_reflected_x:
            x *= -1
        if self.is_reflected_y:
            y *= -1

        x = x + self.x
        y = y + self.y

        return x, y

    def draw(self, ax=None, transform=None, xlim=None, ylim=None):
        """ Draws the feature.

        Parameters:
            ax: plt.Axes (optional)
                Axes on which to draw the feature. If None, will use the current Axes instance.

            transform: matplotlib Transform (optional)
                Transform to apply to the feature.

            xlim: tuple
                (xmin, xmax) to clip x-coordinates.

            ylim: tuple
                (ymin, ymax) to clip y-coordinates.

        Returns:
            plt.Polygon
        """

        if not self.visible:
            return None

        if ax is None:
            ax = plt.gca()

        transform = transform or ax.transData

        patch = self.get_polygon()

        ax.add_patch(patch)
        patch.set_transform(transform)

        # Early exit if no clipping.
        if not (self.clip_xy or xlim or ylim):
            return patch

        # Use either the bounds provided or the feature's own bounds to clip it.
        (xmin, xmax), (ymin, ymax) = self.get_limits()
        xlim = (max(xlim[0], xmin), min(xlim[1], xmax)) if xlim else (xmin, xmax)
        ylim = (max(ylim[0], ymin), min(ylim[1], ymax)) if ylim else (ymin, ymax)

        # Clip based on class attribute.
        if self.clip_xy:
            clip_x, clip_y = self.clip_xy
            clip_x = np.clip(clip_x, *xlim)
            clip_y = np.clip(clip_y, *ylim)

        # Clip based on bounding box of limits.
        else:
            clip_x = [xlim[0], xlim[0], xlim[1], xlim[1]]
            clip_y = [*ylim, *ylim[::-1]]

        clip_polygon = plt.Polygon(list(zip(clip_x, clip_y)), transform=transform)
        patch.set_clip_path(clip_polygon)

        return patch

    def get_limits(self):
        """ Find the outer bounds for the x and y-coordinates of the feature.

        Returns:
            (xmin, xmax), (ymin, ymax)
        """

        if self.clip_xy is None:
            x, y = self.get_polygon_xy()
        else:
            x, y = self.clip_xy

        return (np.min(x), np.max(x)), (np.min(y), np.max(y))


class RinkRectangle(RinkFeature):
    """ A rectangle to be drawn on the rink (typically lines).

    Inherits from RinkFeature.
    """

    def get_centered_xy(self):
        half_length = self.length / 2
        half_width = self.width / 2

        x = np.array([
            -half_length,  # Lower left.
            half_length,  # Lower right.
            half_length,  # Upper right.
            -half_length,  # Upper left.
        ])
        y = np.array([
            -half_width,  # Lower left.
            -half_width,  # Lower right.
            half_width,  # Upper right.
            half_width,  # Upper left.
        ])

        return x, y


class RinkCircle(RinkFeature):
    """ A circle to be drawn on the rink.

    Inherits from RinkFeature.

    When thickness is not 0, the radius attribute is the distance to the outer edge, not the inner.
    """

    def get_centered_xy(self):
        return self.arc_coords(
            center=(0, 0),
            width=self.radius - self.thickness,
            thickness=self.thickness,
            resolution=self.resolution,
        )


class TrapezoidLine(RinkFeature):
    """ A diagonal line typically used for the trapezoid of the goalie's restricted area.

    The x attribute is one end of the line.
    The y attribute is the center of the line.
    The length attribute is the change in the x-coordinate between the ends of the line.
    The width attribute is the change in the y-coordinate between the ends of the line.

    Inherits from RinkFeature.
    """

    def get_centered_xy(self):
        half_thickness = self.thickness / 2

        x = np.array([
            0,  # Inside bottom.
            0,  # Inside top.
            self.length,  # Outside top.
            self.length,  # Outside bottom.
        ])
        y = np.array([
            -half_thickness,  # Inside bottom.
            half_thickness,  # Inside top.
            self.width + half_thickness,  # Outside top.
            self.width - half_thickness,  # Outside bottom.
        ])

        return x, y


class InnerDot(RinkFeature):
    """ The shape inside a faceoff dot.

    Inherits from RinkFeature.

    The length attribute is the distance from the center to the edge of the opening.
    The thickness attribute is the size of the edge of the circle.
    """

    def get_centered_xy(self):
        radius = self.radius - self.thickness
        circle_x, circle_y = self.arc_coords(
            center=(0, 0),
            width=radius,
            resolution=self.resolution,
        )

        half_length = self.length / 2
        mask = (half_length >= circle_x) & (-half_length <= circle_x)

        return circle_x[mask], circle_y[mask]


class FaceoffCircle(RinkFeature):
    """ A circle including hashmarks.

    Inherits from RinkFeature.

    The length attribute is the distance between the inside edges of the hashmarks.
    The width attribute is the length of the hashmarks.
    The radius attribute is the distance to the outer edge, not the inner.
    """

    def get_centered_xy(self):
        circle_x, circle_y = self.arc_coords(
            center=(0, 0),
            width=self.radius - self.thickness,
            thickness=self.thickness,
            resolution=self.resolution,
        )

        inner_circle_x, outer_circle_x = np.reshape(circle_x, (2, -1))
        inner_circle_y, outer_circle_y = np.reshape(circle_y, (2, -1))

        half_length = self.length / 2

        # End of the hashmark.
        hashmark_y = (
            (self.radius ** 2 - half_length ** 2) ** 0.5
            + self.thickness
            + self.width
        )
        hashmark = np.full(outer_circle_y.shape, hashmark_y) * np.sign(outer_circle_y)

        # Replace the y-coordinates on the outer edge of the circle with the end of the hashmark.
        mask = (
            (np.abs(outer_circle_x) >= self.width)
            & (np.abs(outer_circle_x) <= self.width + self.thickness)
        )
        outer_circle_y[mask] = hashmark[mask]

        x = np.concatenate((inner_circle_x, outer_circle_x))
        y = np.concatenate((inner_circle_y, outer_circle_y))

        return x, y


class RinkL(RinkFeature):
    """ L shapes to be drawn on the rink. Typically, used for faceoff lines.

    Inherits from RinkFeature.

    The x and y attributes are the coordinates to the inside corner of the L.
    The length attribute is the size of the vertical line of the L.
    The width attribute is the size of the horizontal line of the L.
    """

    def get_centered_xy(self):
        x = np.array(([
            0,  # Corner of L - left side.
            0,  # Top of L - left side.
            self.thickness * np.sign(self.length),  # Top of L - right side.
            self.thickness * np.sign(self.length),  # Corner of L - right side.
            self.length,  # Right of L - top side.
            self.length,  # Right of L - bottom side.
        ]))

        y = np.array(([
            0,  # Corner of L - left side.
            self.width,  # Top of L - left side.
            self.width,  # Top of L - right side.
            self.thickness * np.sign(self.width),  # Corner of L - right side.
            self.thickness * np.sign(self.width),  # Right of L - top side.
            0,  # Right of L - bottom side.
        ]))

        return x, y


class Crease(RinkFeature):
    """ A goaltender's crease.

    Inherits from RinkFeature.

    Can be used for both the crease (when thickness is 0) and the outline of the crease (when thickness isn't 0).

    Typically a rectangle with an arc at the end, but the rectangle can be removed by setting length to 0.

    The x attribute is the goal line edge of the crease.
    The y attribute is the center of the crease.
    The length attribute is the length of the rectangular portion of the crease.
    The width attribute is the width of the crease, including the thickness of the outline.
    """

    def get_centered_xy(self):
        half_width = self.width / 2

        arc_x, arc_y = self.arc_coords(
            center=(self.length, 0),
            width=self.radius - self.thickness,
            height=half_width - self.thickness,
            thickness=self.thickness,
            theta1=90,
            theta2=-90,
            resolution=self.resolution,
        )

        # The x-coordinates are negative because the crease is opposite
        # of the side of ice its on (eg the crease on the right side of the
        # ice goes to the left).

        # Crease.
        if self.thickness == 0:
            x = np.concatenate((
                [0, 0],
                -arc_x,
                [0, 0],
            ))
            y = np.concatenate((
                [0, half_width],
                arc_y,
                [-half_width, 0],
            ))

        # Crease outline.
        else:
            inner_arc_x, outer_arc_x = np.reshape(arc_x, (2, -1))
            inner_arc_y, outer_arc_y = np.reshape(arc_y, (2, -1))

            x = np.concatenate((
                [0],  # Top inside.
                -inner_arc_x,  # Inside arc.
                [0],  # Bottom inside.
                [0],  # Bottom outside.
                -outer_arc_x,  # Outside arc.
                [0],  # Top outside.
            ))

            y = np.concatenate((
                [half_width - self.thickness],  # Top inside.
                inner_arc_y,  # Inside arc.
                [self.thickness - half_width],  # Bottom inside.
                [-half_width],  # Bottom outside.
                outer_arc_y,  # Outside arc.
                [half_width],  # Top outside.
            ))

        return x, y


class Crossbar(RinkFeature):
    """ The crossbar of a net.

    Inherits from RinkFeature.

    The x attribute is the front edge of the goal line.
    The y attribute is the center of the net.
    The width attribute doesn't include the radius of the bar.
    The radius attribute is used to create an arc at each end of the bar.
    """

    def get_centered_xy(self):
        half_width = self.width / 2

        # Rounded end of the bar.
        arc_x, arc_y = self.arc_coords(
            center=(self.radius, 0),
            width=self.radius,
            theta2=180,
            resolution=self.resolution,
        )

        x = np.concatenate((arc_x, arc_x[::-1]))
        y = np.concatenate((arc_y + half_width, -arc_y[::-1] - half_width))

        return x, y


class Net(RinkFeature):
    """ The netting of a hockey net.

    Inherits from RinkFeature.

    The x attribute is the back edge of the crossbar.
    The y attribute is the center of the net.
    The length attribute is the depth of the net.
    The width attribute is the distance from the end of one post to the end of the other.
    The thickness attribute is the outer width of the net (including the arc at the back).
    The radius attribute is used for the arc at the back of the net.
    """

    def get_centered_xy(self):
        half_width = self.width / 2
        half_outer_width = max(half_width, self.thickness / 2)

        center_x = self.length - self.radius
        center_y = half_outer_width - self.radius

        arc_x, arc_y = self.arc_coords(
            center=(center_x, center_y),
            width=self.radius,
            theta1=180,
            theta2=0,
            resolution=self.resolution,
        )
        tangent_x, tangent_y = self.tangent_point(
            (center_x, center_y),
            self.radius,
            (0, half_width),
        )

        mask = (
            (0 < arc_x)
            & (arc_x > tangent_x)
            & (arc_y > -half_outer_width)
        )
        arc_x = arc_x[mask]
        arc_y = arc_y[mask]

        x = np.concatenate((
            arc_x,
            arc_x[::-1],
            [0, 0],
        ))
        y = np.concatenate((
            arc_y,
            -arc_y[::-1],
            [-half_width, half_width],
        ))

        return x, y


class CircularImage(RinkCircle):
    """ An image to be cropped inside a circle.

    Inherits from RinkFeature.

    If is_constrained is set to True, the image will no longer be circular.

    Additional attributes:
        path: string
            The path from which to read the file for the image.

        rotation: float
            The amount to rotate the image.
    """

    def __init__(self, path, rotation=0, **polygon_kwargs):
        self.path = path
        self.rotation = rotation
        super().__init__(**polygon_kwargs)

    def draw(self, ax=None, transform=None, xlim=None, ylim=None):
        # Early exit if not visible.
        if not self.polygon_kwargs.get("visible", True):
            return None

        if ax is None:
            ax = plt.gca()

        transform = transform or ax.transData

        try:
            image = plt.imread(self.path)

            x = -self.x if self.is_reflected_x else self.x
            y = -self.y if self.is_reflected_y else self.y

            extent = [
                int(x - self.radius), int(x + self.radius),
                int(y - self.radius), int(y + self.radius),
            ]

            im = ax.imshow(image, extent=extent, **self.polygon_kwargs)
            im.set_transform(
                Affine2D().rotate_deg_around(self.x, self.y, self.rotation)
                + transform
            )

            patch = plt.Circle(
                (x, y),
                radius=self.radius,
                transform=transform,
            )
            im.set_clip_path(patch)

            return im

        except Exception as e:
            print(e)

            return None


class LowerInwardArcRectangle(RinkFeature):
    """ A rectangle with one rounded edge. This can be used, for example, at center ice when the red line
    should not extend all the way across the rink, but stop at the faceoff circle. The rounded edge will
    arc inside the bottom of the rectangle (or top if reflected).

    The x attribute is the center of the rectangle.
    The y attribute is the bottom of the rectangle.
    The length attribute is the bottom to top distance including the radius.

    Inherits from RinkFeature.
    """

    def get_centered_xy(self):
        arc_x, arc_y = self.arc_coords(
            center=(0, 0),
            width=self.radius,
            theta2=180,
            resolution=self.resolution,
        )

        # Only use coordinates that are within the left-to-right space of the feature.
        half_length = self.length / 2
        mask = (arc_x >= -half_length) & (arc_x <= half_length)
        arc_x = arc_x[mask]
        arc_y = arc_y[mask]

        x = np.concatenate([arc_x, [-half_length, half_length]])
        y = np.concatenate([arc_y, [self.width, self.width]])

        return x, y


class RoundedRectangle(RinkFeature):
    """ A rounded rectangle to be drawn on the rink (typically, the boards).

    Inherits from RinkFeature.

    The radius attribute is the radius of the arc for rounded edges. Inappropriate values will lead to
    unusual shapes.
    """

    def get_centered_xy(self):
        end_x = self.length / 2
        end_y = self.width / 2

        center_x = end_x - self.radius
        center_y = end_y - self.radius

        arc_x, arc_y = self.arc_coords(
            center=(center_x, center_y),
            width=self.radius,
            thickness=self.thickness,
            theta1=90,
            theta2=0,
        )

        if self.thickness == 0:
            inner_arc_x = arc_x
            outer_arc_x = np.array([])
            inner_arc_y = arc_y
            outer_arc_y = np.array([])
        else:
            inner_arc_x, outer_arc_x = np.reshape(arc_x, (2, -1))
            inner_arc_y, outer_arc_y = np.reshape(arc_y, (2, -1))

        x = np.concatenate((
            inner_arc_x,  # Inside top right corner.
            inner_arc_x[::-1],  # Inside bottom right corner.
            -inner_arc_x,  # Inside bottom left corner.
            -inner_arc_x[::-1],  # Inside top left corner.
            inner_arc_x[:1],  # Start of inside top left corner.
            outer_arc_x[-1:],  # End of inside top left corner.
            -outer_arc_x[::-1],  # Outside top left corner.
            -outer_arc_x,  # Outside bottom left corner.
            outer_arc_x[::-1],  # Outside bottom right corner.
            outer_arc_x,  # Outside top right corner.
        ))

        y = np.concatenate((
            inner_arc_y,  # Inside corner.
            -inner_arc_y[::-1],  # Inside bottom right corner.
            -inner_arc_y,  # Inside bottom left corner.
            inner_arc_y[::-1],  # Inside top left corner.
            inner_arc_y[:1],  # Start of inside top left corner.
            outer_arc_y[-1:],  # End of inside top left corner.
            outer_arc_y[::-1],  # Outside top left corner.
            -outer_arc_y,  # Outside bottom left corner.
            -outer_arc_y[::-1],  # Outside bottom right corner.
            outer_arc_y,  # Outside top right corner.
        ))

        return x, y
