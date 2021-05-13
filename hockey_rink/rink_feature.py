""" Module containing all provided rink features as well as the abstract base class.

Currently available features are:
    Boards
    BoardsConstraint
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
"""


__all__ = ["RinkFeature", "Boards", "BoardsConstraint", "RinkRectangle",
           "RinkCircle", "TrapezoidLine", "InnerDot", "FaceoffCircle",
           "RinkL", "Crease", "Crossbar", "Net", "CircularImage"]


from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np


class RinkFeature(ABC):
    """ Abstract base class for features of a rink to draw.

    Attributes:
        x: float

        y: float

        length: float
            Typically, the size of the feature from left to right.

        width: float
            Typically, the size of the feature from bottom to top.

        thickness: float

        radius: float

        resolution: int
            The number of coordinates used in creating arcs.

        x_reflection: {-1, 1}
            If -1, reflects all x-coordinates.
            If 1, leaves x-coordinates as they are.

        y_reflection: {-1, 1}
            If -1, reflects all y-coordinates.
            If 1, leaves y-coordinates as they are.

        is_constrained: bool
            Indicates whether or not the feature is constrained to remain inside the boards.

        visible: bool

        polygon_kwargs: additional keyword arguments
            Any additional arguments to be passed to matplotlib's Polygon.
    """

    def __init__(self, x=0, y=0, length=0, width=0, thickness=0, radius=0,
                 resolution=500, is_reflected_x=False, is_reflected_y=False,
                 is_constrained=True, visible=True, **polygon_kwargs):
        """ Initialize attributes.

        Parameters:
            x: float; default: 0

            y: float; default: 0

            length: float; default: 0

            width: float; default: 0

            thickness: float; default: 0

            radius: float; default: 0

            resolution: int; default: 500

            is_reflected_x: bool; default: False
                Indicates if x-coordinates are to be reflected.

            y_reflection: bool; default: False
                Indicates if y-coordinates are to be reflected.

            is_constrained: bool; default: True

            visible: bool; default: True

            polygon_kwargs: additional keyword arguments; optional
        """

        self.x = x
        self.y = y
        self.length = length
        self.width = width
        self.thickness = thickness
        self.radius = radius
        self.resolution = resolution
        self.x_reflection = -1 if is_reflected_x else 1
        self.y_reflection = -1 if is_reflected_y else 1
        self.is_constrained = is_constrained
        self.visible = visible
        self.polygon_kwargs = polygon_kwargs

    @abstractmethod
    def _get_centered_xy(self):
        """ Abstract method to return the x and y-coordinates necessary to create a Polygon of the feature
        were it to be recorded at (0, 0).

        The actual coordinates used are left to get_polygon_xy.

        Returns:
            numpy ndarray, numpy ndarray
        """

        pass

    @staticmethod
    def arc_coords(center, width, height=None, thickness=0, theta1=0, theta2=360, resolution=500):
        """" Return the coordinates of an arc.

        Adapted from: https://stackoverflow.com/questions/30642391/how-to-draw-a-filled-arc-in-matplotlib

        Parameters:
            center: tuple
                The center point of the arc.

            width: float

            height: float; optional
                If not provided, height will be equal to width, creating a circular arc.

            thickness: float; default: 0
                If not 0, will arcs: one with the width and height provided and one with the width plus the
                thickness and the height plus the thickness.

            theta1: float; default: 0
                Degree at which to start the arc.

                0 corresponds to directly to the right of center, 90 to directly above, etc.

            theta2: float; default: 360
                Degree at which to end the arc.

                0 corresponds to directly to the right of center, 90 to directly above, etc.

            resolution: int; default: 500
                The number of coordinates used in creating the arc.

        Returns:
            numpy ndarray, numpy ndarray
        """

        height = width if height is None else height

        theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
        x = width * np.cos(theta) + center[0]
        y = height * np.sin(theta) + center[1]

        if thickness > 0:
            theta = np.linspace(np.radians(theta2), np.radians(theta1), resolution)
            x = np.concatenate((x, (width + thickness) * np.cos(theta) + center[0]))
            y = np.concatenate((y, (height + thickness) * np.sin(theta) + center[1]))

        return x, y

    @staticmethod
    def tangent_point(center, radius, point):
        """ Return the coordinates on a circle that are tangent to a point outside the circle.

        Adapted from: https://stackoverflow.com/a/49987361

        Parameters:
            center: tuple
                The center point of the circle.

            radius: float

            point: tuple
                The coordinates of the point outside the circle.

        Returns:
            float, float
        """

        dx = point[0] - center[0]
        dy = point[1] - center[1]
        center_to_point = (dx ** 2 + dy ** 2) ** 0.5
        theta = np.arccos(radius / center_to_point)
        angle = np.arctan2(dy, dx) - theta * np.sign(point[1])

        return center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)

    def get_polygon(self):
        """ Return the matplotlib Polygon representing the feature. """

        polygon_x, polygon_y = self.get_polygon_xy()

        return plt.Polygon(tuple(zip(polygon_x, polygon_y)), visible=self.visible, **self.polygon_kwargs)

    def get_polygon_xy(self):
        """ Return numpy ndarray, numpy ndarray of the x,y-coordinates necessary for creating the
        matplotlib Polygon representing the feature.
        """

        x, y = self._get_centered_xy()
        x = x * self.x_reflection + self.x
        y = y * self.y_reflection + self.y

        return x, y

    def draw(self, ax, transform=None):
        """ Draw the feature.

        Parameters:
            ax: matplotlib Axes
                Axes in which to draw the feature.

            transform: matplotlib Transform; optional
                Transform to apply to the feature.

        Returns:
            matplotlib Polygon
        """

        transform = transform or ax.transData

        patch = self.get_polygon()
        ax.add_patch(patch)
        patch.set_transform(transform)

        return patch


class Boards(RinkFeature):
    """ One quarter of a rink's boards.

    length and width refer to the full ice surface, not this specific quadrant.
    radius is for the arc in the corner boards.

    See RinkFeature for full documentation.
    """

    def _get_centered_xy(self):
        end_x = self.length / 2
        end_y = self.width / 2

        center_x = end_x - self.radius
        center_y = end_y - self.radius

        arc_x, arc_y = self.arc_coords((center_x, center_y), width=self.radius,
                                       thickness=self.thickness, theta1=90, theta2=0)

        inner_arc_x, outer_arc_x = np.reshape(arc_x, (2, -1))
        inner_arc_y, outer_arc_y = np.reshape(arc_y, (2, -1))

        board_x = np.concatenate((
            [0],    # inside center ice
            inner_arc_x,    # inside corner
            [end_x],    # inside end boards
            [end_x + self.thickness],    # outside end boards
            outer_arc_x,    # outside corner
            [0],    # outside center ice
        ))

        board_y = np.concatenate((
            [end_y],
            inner_arc_y,
            [0],
            [0],
            outer_arc_y,
            [end_y + self.thickness],
        ))

        return board_x, board_y


class BoardsConstraint(RinkFeature):
    """ The inside edge of the boards.  To be used to constrain other features to being inside the rink.

    Unlike boards, is used for the entire ice surface.

    radius is for the arc in the corner boards.

    See RinkFeature for full documentation.
    """

    def _get_centered_xy(self):
        end_x = self.length / 2
        end_y = self.width / 2

        center_x = end_x - self.radius
        center_y = end_y - self.radius

        arc_x, arc_y = self.arc_coords((center_x, center_y), width=self.radius,
                                       theta1=90, theta2=0, resolution=self.resolution)

        board_x = np.concatenate((
            arc_x,    # top right corner
            arc_x[::-1],    # bottom right corner
            -arc_x,    # bottom left corner
            -arc_x[::-1],    # top left corner
        ))

        board_y = np.concatenate((
            arc_y,
            -arc_y[::-1],
            -arc_y,
            arc_y[::-1],
        ))

        return board_x, board_y


class RinkRectangle(RinkFeature):
    """ A rectangle to be drawn on the rink (typically lines).

    x and y refer to the center of the rectangle.
    length is always left to right.
    width is always bottom to top.

    See RinkFeature for full documentation.
    """

    def _get_centered_xy(self):
        half_length = self.length / 2
        half_width = self.width / 2
        x = np.array([-half_length, half_length, half_length, -half_length])
        y = np.array([-half_width, -half_width, half_width, half_width])

        return x, y


class RinkCircle(RinkFeature):
    """ A circle to be drawn on the rink.

    x and y refer to the center of the circle.
    When including thickness, radius is distance to the outer edge, not the inner.

    See RinkFeature for full documentation.
    """

    def _get_centered_xy(self):
        return self.arc_coords((0, 0), width=self.radius - self.thickness,
                               thickness=self.thickness, resolution=self.resolution)


class TrapezoidLine(RinkFeature):
    """ One line of a trapezoid (typically for the goalie's restricted area).

    length is distance from left to right.
    width is distance from bottom to top.

    See RinkFeature for full documentation.
    """

    def _get_centered_xy(self):
        half_thickness = self.thickness / 2

        x = np.array([0, 0, self.length, self.length])
        y = np.array([-half_thickness, half_thickness,
                      self.width + half_thickness, self.width - half_thickness])

        return x, y


class InnerDot(RinkFeature):
    """ The shape inside a faceoff dot.

    x and y refer to the center of the dot.
    length is from the center to the edge of the opening.
    thickness is the size of the edge of the circle.

    See RinkFeature for full documentation.
    """

    def _get_centered_xy(self):
        radius = self.radius - self.thickness
        circle_x, circle_y = self.arc_coords((0, 0), width=radius,
                                             resolution=self.resolution)

        half_length = self.length / 2
        mask = (half_length >= circle_x) & (-half_length <= circle_x)

        return circle_x[mask], circle_y[mask]


class FaceoffCircle(RinkFeature):
    """ A circle including hashmarks.

    x and y refer to the center of the circle.
    length is distance between inside edges of the hashmarks.
    width is length of the hashmarks.
    radius is distance to the outer edge, not the inner.

    See RinkFeature for full documentation.
    """

    def _get_centered_xy(self):
        circle_x, circle_y = self.arc_coords((0, 0), width=self.radius - self.thickness,
                                             thickness=self.thickness, resolution=self.resolution)

        inner_circle_x, outer_circle_x = np.reshape(circle_x, (2, -1))
        inner_circle_y, outer_circle_y = np.reshape(circle_y, (2, -1))

        half_length = self.length / 2

        # end of the hashmark
        hashmark_y = (self.radius ** 2 - half_length ** 2) ** 0.5 + self.thickness + self.width
        hashmark = np.full(outer_circle_y.shape, hashmark_y) * np.sign(outer_circle_y)

        # replace the y coordinates on the outer edge of the circle with the end of the hashmark
        mask = ((np.abs(outer_circle_x) >= self.width)
                & (np.abs(outer_circle_x) <= self.width + self.thickness))
        outer_circle_y[mask] = hashmark[mask]

        x = np.concatenate((inner_circle_x, outer_circle_x))
        y = np.concatenate((inner_circle_y, outer_circle_y))

        return x, y


class RinkL(RinkFeature):
    """ L shapes to be drawn on the rink (typically for faceoff lines).

    x and y refer to the inside corner of the L.
    length is the | of the L.
    width is the _ of the L.

    See RinkFeature for full documentation.
    """

    def _get_centered_xy(self):
        # inside corner -> top of L -> outside corner -> right of L
        x = np.array(([0, 0, self.thickness * np.sign(self.length),
                       self.thickness * np.sign(self.length), self.length, self.length]))
        y = np.array(([0, self.width, self.width, self.thickness * np.sign(self.width),
                       self.thickness * np.sign(self.width), 0]))

        return x, y


class Crease(RinkFeature):
    """ A goaltender's crease.

    Used for both the crease (when thickness is 0) and the outline of the crease (when thickness isn't 0).

    Typically a rectangle with an arc at the end, but the rectangle can be removed by setting length to 0.

    x is goal line edge of crease.
    y is center of crease.
    length is the length of the rectangular portion of the crease.
    width is the width of the crease including the thickness of the outline.

    See RinkFeature for full documentation.
    """

    def _get_centered_xy(self):
        half_width = self.width / 2

        arc_x, arc_y = self.arc_coords(center=(self.length, 0),
                                       width=self.radius - self.thickness,
                                       height=half_width - self.thickness,
                                       thickness=self.thickness,
                                       theta1=90, theta2=0,
                                       resolution=self.resolution)

        if self.thickness == 0:    # crease
            x = np.concatenate(([0, 0], -arc_x))
            y = np.concatenate(([0, half_width], arc_y))
        else:    # crease outline
            x = np.concatenate(([0], -arc_x, [0]))
            y = np.concatenate(([half_width - self.thickness], arc_y, [half_width]))

        return x, y


class Crossbar(RinkFeature):
    """ The crossbar of a net.

    x is the front edge of the goal line.
    y is the center of the net.
    width doesn't include the radius of the bar.
    radius is used to create an arc at each end of the bar.

    See RinkFeature for full documentation.
    """

    def _get_centered_xy(self):
        half_width = self.width / 2

        # rounded end of the bar
        arc_x, arc_y = self.arc_coords((self.radius, 0), width=self.radius,
                                       theta2=180, resolution=self.resolution)

        x = np.concatenate((arc_x, arc_x[::-1]))
        y = np.concatenate((arc_y + half_width, -arc_y[::-1] - half_width))

        return x, y


class Net(RinkFeature):
    """ The netting of a hockey net.

    x is the back edge of the crossbar.
    y is the center of the net.
    length is the depth of the net.
    width is from the end of one post to the end of the other.
    thickness is the outer width (including the arc at the back).
    radius is for the arc at the back of the net.

    See RinkFeature for full documentation.
    """

    def _get_centered_xy(self):
        half_width = self.width / 2
        half_outer_width = max(half_width, self.thickness / 2)

        center_x = self.length - self.radius
        center_y = half_outer_width - self.radius

        arc_x, arc_y = self.arc_coords((center_x, center_y), width=self.radius,
                                       theta1=180, theta2=0, resolution=self.resolution)
        tangent_x, tangent_y = self.tangent_point((center_x, center_y), self.radius, (0, half_width))

        mask = (0 < arc_x) & (arc_x > tangent_x) & (arc_y > -half_outer_width)
        arc_x = arc_x[mask]
        arc_y = arc_y[mask]

        x = np.concatenate((arc_x, arc_x[::-1], [0, 0]))
        y = np.concatenate((arc_y, -arc_y[::-1], [-half_width, half_width]))

        return x, y


class CircularImage(RinkCircle):
    """ An image to be cropped inside a circle.

    Requires a path variable of the image file to be read.
    x and y refer to the center of the circle.
    If is_constrained is set to False, the image will no longer be circular.

    See RinkFeature for full documentation.
    """

    def __init__(self, path, rotation=0, is_constrained=False, **polygon_kwargs):
        self.path = path
        self.rotation = rotation
        super().__init__(is_constrained=is_constrained, **polygon_kwargs)

    def draw(self, ax, transform=None):
        """ Draw the feature.

        Parameters:
            ax: matplotlib Axes
                Axes in which to draw the feature.

            transform: matplotlib Transform; optional
                Transform to apply to the feature.

        Returns:
            matplotlib AxesImage
        """

        if not self.visible:
            return None

        transform = transform or ax.transData

        try:
            image = plt.imread(self.path)

            x = self.x * self.x_reflection
            y = self.y * self.y_reflection

            extent = [int(x - self.radius), int(x + self.radius),
                      int(y - self.radius), int(y + self.radius)]

            im = ax.imshow(image, extent=extent, **self.polygon_kwargs)
            im.set_transform(Affine2D().rotate_deg_around(self.x, self.y, self.rotation)
                             + transform)

            patch = plt.Circle((x, y), radius=self.radius, transform=transform)
            im.set_clip_path(patch)

            return im

        except Exception as e:
            print(e)

            return None
