from hockey_rink.base_features import RinkFeature
import matplotlib.pyplot as plt
import numpy as np


class Boards(RinkFeature):
    """ The boards around the rink.

    Inherits from RinkFeature.

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
        color="black", zorder=100,
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
            color: color (default="black")
            zorder: float (default=100)
            polygon_kwargs: dict (optional)
        """

        polygon_kwargs["color"] = color
        polygon_kwargs["zorder"] = zorder
        super().__init__(
            x, y,
            length, width, thickness,
            radius, resolution,
            is_reflected_x, is_reflected_y,
            is_constrained,
            **polygon_kwargs,
        )

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

        # Still need coordinates when thickness is 0. Reversing the inner arc coordinates will result in the
        # boards not being drawn while still having coordinates.
        if self.thickness == 0:
            inner_arc_x = arc_x
            outer_arc_x = arc_x[::-1]
            inner_arc_y = arc_y
            outer_arc_y = arc_y[::-1]
        else:
            inner_arc_x, outer_arc_x = np.reshape(arc_x, (2, -1))
            inner_arc_y, outer_arc_y = np.reshape(arc_y, (2, -1))

        board_x = np.concatenate((
            inner_arc_x,    # Inside top right corner.
            inner_arc_x[::-1],    # Inside bottom right corner.
            -inner_arc_x,    # Inside bottom left corner.
            -inner_arc_x[::-1],    # Inside top left corner.
            inner_arc_x[:1],    # Start of inside top left corner.
            outer_arc_x[-1:],    # End of inside top left corner.
            -outer_arc_x[::-1],    # Outside top left corner.
            -outer_arc_x,    # Outside bottom left corner.
            outer_arc_x[::-1],    # Outside bottom right corner.
            outer_arc_x,    # Outside top right corner.
        ))

        board_y = np.concatenate((
            inner_arc_y,    # Inside corner.
            -inner_arc_y[::-1],    # Inside bottom right corner.
            -inner_arc_y,    # Inside bottom left corner.
            inner_arc_y[::-1],    # Inside top left corner.
            inner_arc_y[:1],    # Start of inside top left corner.
            outer_arc_y[-1:],    # End of inside top left corner.
            outer_arc_y[::-1],    # Outside top left corner.
            -outer_arc_y,    # Outside bottom left corner.
            -outer_arc_y[::-1],    # Outside bottom right corner.
            outer_arc_y,    # Outside top right corner.
        ))

        return board_x, board_y

    def get_constraint_xy(self):
        """ Determines the x and y-coordinates necessary for bounding the rink by the boards. Only the inner arc of
        the boards is needed for creating the bound.

        Returns:
            np.array, np.array
        """

        x, y = self.get_polygon_xy()

        # Only want the inside edge of the boards.
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
