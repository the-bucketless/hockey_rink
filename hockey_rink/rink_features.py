""" Module containing features to draw on the rink.

Available features are:
    Boards
    RinkImage
    CircularImage
"""


__all__ = [
    "Boards",
    "RinkImage",
    "CircularImage",
]

from hockey_rink.base_features import RinkRectangle, RoundedRectangle
import matplotlib.pyplot as plt
import numpy as np


class Boards(RoundedRectangle):
    """ The boards around the rink.

    Inherits from RoundedRectangle.

    The length and width attributes are the size of the ice surface, not including the thickness of the boards. When
    thickness is 0, the boards will not be drawn.

    The radius attribute is the radius of the arc for the corner of the boards. Inappropriate values will lead to
    unusual shapes.
    """

    def __init__(
        self,
        x=0, y=0,
        length=200, width=85, thickness=1,
        radius=28, resolution=500,
        is_reflected_x=False, is_reflected_y=False,
        visible=True, color="black", zorder=100,
        clip_xy=None,
        **polygon_kwargs,
    ):
        """ Initialize attributes.

        Parameters:
            x: float (default=0)
            y: float (default=0)
            length: float (default=200)
            width: float (default=85)
            thickness: float (default=1)
            radius: float (default=28)
            resolution: int (default=500)
            is_reflected_x: bool (default=False)
            is_reflected_y: bool (default=False)
            visible: bool (default=True)
            color: color (default="black")
            zorder: float (default=100)
            clip_xy: (np.array, np.array) (optional)
            polygon_kwargs: dict (optional)
        """

        # Avoid drawing boards when thickness is 0.
        if thickness == 0:
            polygon_kwargs["fill"] = polygon_kwargs.get("fill", False)
            polygon_kwargs["linewidth"] = polygon_kwargs.get("linewidth", 0)

        super().__init__(
            x, y,
            length, width, thickness,
            radius, resolution,
            is_reflected_x, is_reflected_y,
            visible, color, zorder,
            clip_xy,
            **polygon_kwargs,
        )

    def get_xy_for_clip(self):
        """ Determines the x and y-coordinates necessary for bounding the rink by the boards. Only the inner arc of
        the boards is needed for creating the bound.

        Returns:
            np.array, np.array
        """

        x, y = self.get_polygon_xy()

        # Only want the inside edge of the boards.
        if self.thickness:
            x = x[:len(x) // 2]
            y = y[:len(y) // 2]

        return x, y


class RinkImage(RinkRectangle):
    """ An image to be drawn on the rink.

    Inherits from RinkRectangle.

    The length and width attributes control the image's extent, unless it is otherwise specified.

    Additional attributes:
        image: np.array
            The image to be drawn.
    """

    def __init__(
        self,
        x=0, y=0,
        length=None, width=None, thickness=None,
        radius=None, resolution=None,
        is_reflected_x=False, is_reflected_y=False,
        visible=True, color=None, zorder=None,
        clip_xy=None,
        image=None, image_path=None,
        **polygon_kwargs,
    ):
        """ Initialize attributes.

        Parameters:
            x: float (default=0)
            y: float (default=0)

            length: float (optional)
                If not included, will be inferred by the size of the image.

            width: float (optional)
                If not included, will be inferred by the size of the image.

            thickness: Ignored
            radius: Ignored
            resolution: Ignored
            is_reflected_x: bool (default=False)
            is_reflected_y: bool (default=False)
            visible: bool (default=True)
            color: color (optional)
            zorder: float (optional)
            clip_xy: (np.array, np.array) (optional)
            image: np.array (optional)
            image_path: string (optional)
            polygon_kwargs: dict (optional)

        Raises:
            Exception if neither of image and image_path are specified.
        """

        if image is None and image_path is None:
            raise Exception("One of image and image_path must be specified when creating RinkImage.")

        self.image = plt.imread(image_path) if image is None else image

        image_height, image_width, *_ = self.image.shape
        if length is None:
            length = image_width
        if width is None:
            width = image_height

        super().__init__(
            x, y,
            length, width, thickness,
            radius, resolution,
            is_reflected_x, is_reflected_y,
            visible, color, zorder,
            clip_xy,
            **polygon_kwargs,
        )

    def draw(self, ax=None, transform=None, xlim=None, ylim=None):
        if not self.visible:
            return None

        if ax is None:
            ax = plt.gca()

        # Create copy to avoid mutating with pops.
        image_kwargs = dict(self.polygon_kwargs)

        transform = transform or ax.transData
        image_transform = image_kwargs.pop("transform", None)
        image_transform = transform if image_transform is None else image_transform + transform

        # If extent not provided, use length and width of the image to calculate it.
        if "extent" in image_kwargs:
            extent = image_kwargs.pop("extent")
        else:
            polygon_x, polygon_y = self.get_polygon_xy()
            extent = [
                np.min(polygon_x), np.max(polygon_x),
                np.min(polygon_y), np.max(polygon_y),
            ]

        img = ax.imshow(self.image, extent=extent, transform=image_transform, **image_kwargs)

        if self.clip_xy is not None:
            img = self._clip_patch(img, transform, xlim, ylim)

        return img


class CircularImage(RinkImage):
    """ An image to be cropped inside a circle.

    Inherits from RinkImage.

    The size of the circle will be set based on the radius attribute.
    """

    def __init__(
        self,
        x=0, y=0,
        length=None, width=None, thickness=None,
        radius=0, resolution=500,
        is_reflected_x=False, is_reflected_y=False,
        visible=True, color=None, zorder=None,
        clip_xy=None,
        image=None, image_path=None,
        **polygon_kwargs,
    ):
        """ Initialize attributes.

        Parameters:
            x: float (default=0)
            y: float (default=0)

            length: float (optional)
                If not included, will be set to twice the size of the radius.

            width: float (optional)
                If not included, will be set to twice the size of the radius.

            thickness: Ignored

            radius: float (default=0)
                If left at 0, image will not be drawn.

            resolution: float (default=500)
            is_reflected_x: bool (default=False)
            is_reflected_y: bool (default=False)
            visible: bool (default=True)
            color: color (optional)
            zorder: float (optional)
            clip_xy: Ignored
            image: np.array (optional)
            image_path: string (optional)
            polygon_kwargs: dict (optional)

        Raises:
            Exception if neither of image and image_path are specified.
        """

        if length is None:
            length = radius * 2
        if width is None:
            width = radius * 2

        super().__init__(
            x, y,
            length, width, thickness,
            radius, resolution,
            is_reflected_x, is_reflected_y,
            visible, color, zorder,
            clip_xy,
            image, image_path,
            **polygon_kwargs,
        )

        # Create the clipping coordinates based on the radius.
        clip_x, clip_y = self.arc_coords(
            center=(0, 0),
            width=self.radius,
            resolution=self.resolution,
        )
        self.clip_xy = self._convert_xy(clip_x, clip_y)
