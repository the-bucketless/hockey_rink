from hockey_rink.base_features import RoundedRectangle
import matplotlib.pyplot as plt


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
        is_constrained=False,
        visible=True, color="black", zorder=100,
        rink=None,
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
            is_constrained: bool (default=False)
            visible: bool (default=True)
            color: color (default="black")
            zorder: float (default=100)
            rink: Rink (optional)
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
            is_constrained,
            visible, color, zorder,
            rink,
            **polygon_kwargs,
        )

    def get_constraint_xy(self):
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

    def get_constraint(self):
        """ Creates a Polygon for use to constrain other objects within the inner edge of the boards.

        Returns:
            plt.Polygon
        """

        x, y = self.get_constraint_xy()
        return plt.Polygon(tuple(zip(x, y)), visible=False)
