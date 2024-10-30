import matplotlib.pyplot as plt
import numpy as np


class WavyArrow:
    """ An arrow with a sine wave as a shaft.

    Attributes will only be computed when the class is called with all relevant parameters.
    """

    def __init__(self):
        self.coords = {}

    def __call__(
        self,
        x, y, dx, dy,
        n_waves=None, wave_frequency=0.25, wave_height=1, resolution=500,
        stem_length=1,
        has_left_head=False, has_right_head=True,
        head_length=3, head_width=4, length_includes_head=True, is_closed=True,
        shaft_kw=None, head_kw=None, left_head_kw=None, right_head_kw=None,
        ax=None,
        **kwargs,
    ):
        """ Plot the arrow.

        The arrowheads and shaft are plotted separately. The shaft makes use of plt.plot() while the arrowheads will
        use either plt.plot() or plt.fill() depending on whether the head is closed.

        Parameters:
            x: float
                The x-coordinate of the base of the arrow.

            y: float
                The y-coordinate of the base of the arrow.

            dx: float
                The length of the arrow in the x direction.

            dy: float
                The length of the arrow in the y direction.

            n_waves: float (optional)
                The number of full sine waves in the arrow. If not provided, will be calculated using wave_frequency.

            wave_frequency: float (default=0.25)
                The number of full waves per 1 unit in plotting coordinates.
                Will be rounded to the nearest half-wave.
                Ignored if n_waves is not None.

            wave_height: float (default=1)
                The height of the crest of the wave.
                The full wave height (crest and trough) will be twice this value.

            resolution: int (default=500)
                The number of coordinates used in creating the wave.

            stem_length: float (default=1)
                The length of the stem(s) connecting the arrowhead(s) and the wave.
                The stem is used to connect the wave at the middle of the arrowhead.

            has_left_head: bool (default=False)
                Whether to include an arrowhead on the left side (x,y) of the arrow.

            has_right_head: bool (default=True)
                Whether to include an arrowhead on the right side (x + dx, y + dy) of the arrow.

            head_length: float (default=3)
                The length of the arrowhead (shaft to tip).

            head_width: float (default=4)
                The width of the base of the arrowhead.

            length_includes_head: bool (default=True)
                Whether the length of the arrow includes the arrowhead.

            is_closed: bool (default=True)
                Whether the arrowhead is closed or not.
                    True: -|>
                    False: ->

            shaft_kw: dict (optional)
                Additional keyword arguments to be provided to the .plot() function for the shaft of the arrow.

            head_kw: dict (optional)
                Additional keyword arguments to be provided to the .plot() or .fill() function for any arrowheads.

            left_head_kw: dict (optional)
                Additional keyword arguments to be provided to the .plot() or .fill() function specifically for the
                left arrowhead. Will supersede head_kw.

            right_head_kw: dict (optional)
                Additional keyword arguments to be provided to the .plot() or .fill() function specifically for the
                right arrowhead. Will supersede head_kw.

            ax: matplotlib Axes (optional)
                If not provided, will use the currently active Axes.

            kwargs: Any other properties that can be provided to the plotting functions for both the shaft and the
                arrowheads.

        Returns:
            self
        """

        self._initialize(
            x, y, dx, dy,
            n_waves, wave_frequency, wave_height, resolution,
            stem_length,
            has_left_head, has_right_head,
            head_length, head_width, length_includes_head, is_closed,
            shaft_kw, head_kw, left_head_kw, right_head_kw,
        )

        if ax is None:
            ax = plt.gca()

        shaft_x, shaft_y = np.array([]), np.array([])
        for feature in ("left_stem", "wave", "right_stem"):
            feature_x, feature_y = self.coords[feature]
            shaft_x = np.concatenate([shaft_x, feature_x])
            shaft_y = np.concatenate([shaft_y, feature_y])

        ax.plot(shaft_x, shaft_y, **{**kwargs, **self.shaft_kw})

        head_fn = ax.fill if self.is_closed else ax.plot

        for has_head, head, kw in zip(
                [self.has_left_head, self.has_right_head],
                ["left_head", "right_head"],
                [self.left_head_kw, self.right_head_kw],
        ):
            if not has_head:
                continue

            if not self.is_closed:
                kw = {k: v for k, v in kw.items() if k not in ("edgecolor", "facecolor")}

            head_fn(*self.coords[head], **{**kwargs, **kw})

        return self

    def _initialize(
        self,
        x, y, dx, dy,
        n_waves=None, wave_frequency=0.25, wave_height=1, resolution=500,
        stem_length=1,
        has_left_head=False, has_right_head=True,
        head_length=3, head_width=4, length_includes_head=True, is_closed=True,
        shaft_kw=None, head_kw=None, left_head_kw=None, right_head_kw=None,
    ):
        """ Initialize all class attributes.

        By default, arrows are black and the arrowheads have a slightly higher zorder than the shafts. This is to
        ensure that, when using thicker lines, the arrowhead still appears on top of the shaft.

        Parameters:
            x: float
                The x-coordinate of the base of the arrow.

            y: float
                The y-coordinate of the base of the arrow.

            dx: float
                The length of the arrow in the x direction.

            dy: float
                The length of the arrow in the y direction.

            n_waves: float (optional)
                The number of full sine waves in the arrow. If not provided, will be calculated using wave_frequency.

            wave_frequency: float (default=0.25)
                The number of full waves per 1 unit in plotting coordinates.
                Will be rounded to the nearest half-wave.
                Ignored if n_waves is not None.

            wave_height: float (default=1)
                The height of the crest of the wave.
                The full wave height (crest and trough) will be twice this value.

            resolution: int (default=500)
                The number of coordinates used in creating the wave.

            stem_length: float (default=1)
                The length of the stem(s) connecting the arrowhead(s) and the wave.
                The stem is used to connect the wave at the middle of the arrowhead.

            has_left_head: bool (default=False)
                Whether to include an arrowhead on the left side (x,y) of the arrow.

            has_right_head: bool (default=True)
                Whether to include an arrowhead on the right side (x + dx, y + dy) of the arrow.

            head_length: float (default=3)
                The length of the arrowhead (shaft to tip).

            head_width: float (default=4)
                The width of the base of the arrowhead.

            length_includes_head: bool (default=True)
                Whether the length of the arrow includes the arrowhead.

            is_closed: bool (default=True)
                Whether the arrowhead is closed or not.
                    True: -|>
                    False: ->

            shaft_kw: dict (optional)
                Additional keyword arguments to be provided to the .plot() function for the shaft of the arrow.

            head_kw: dict (optional)
                Additional keyword arguments to be provided to the .plot() or .fill() function for any arrowheads.

            left_head_kw: dict (optional)
                Additional keyword arguments to be provided to the .plot() or .fill() function specifically for the
                left arrowhead. Will supersede head_kw.

            right_head_kw: dict (optional)
                Additional keyword arguments to be provided to the .plot() or .fill() function specifically for the
                right arrowhead. Will supersede head_kw.
        """

        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy

        self.n_waves = n_waves
        self.wave_frequency = wave_frequency
        self.wave_height = wave_height
        self.resolution = resolution

        self.stem_length = stem_length

        self.has_left_head = has_left_head
        self.has_right_head = has_right_head
        self.head_length = head_length
        self.head_width = head_width
        self.length_includes_head = length_includes_head or not is_closed
        self.is_closed = is_closed

        self.shaft_kw = shaft_kw or {}
        self.head_kw = head_kw or {}
        self.left_head_kw = left_head_kw or {}
        self.right_head_kw = right_head_kw or {}

        shaft_defaults = {"color": "black", "zorder": 1}
        self.shaft_kw = {**shaft_defaults, **self.shaft_kw}

        head_defaults = {"zorder": self.shaft_kw["zorder"] + 1e-6}

        self.left_head_kw = {**head_defaults, **self.head_kw, **self.left_head_kw}
        self.right_head_kw = {**head_defaults, **self.head_kw, **self.right_head_kw}

        self.left_head_kw = self._update_head_color(self.left_head_kw)
        self.right_head_kw = self._update_head_color(self.right_head_kw)

        self.n_heads = self.has_left_head + self.has_right_head

        self._set_xy()

    def _update_head_color(self, head_kw):
        """ Update keyword arguments for head colors, defaulting to black when not provided. """

        if not self.is_closed and "color" not in head_kw:
            head_kw["color"] = head_kw.get("facecolor", "black")
            return head_kw

        if "color" not in head_kw and "facecolor" not in head_kw:
            head_kw["color"] = "black"

        return head_kw

    def _set_xy(self):
        """ Set the coordinates for each of the elements of the arrow (heads, stems, and wave).
        Initially, sets the coordinates as if the arrow was going straight to the right from (0, 0).
        The coordinates are then rotated and translated into place.
        """

        self.arrow_length = np.hypot(self.dx, self.dy)

        self._initialize_heads_xy()
        self._initialize_stems_xy()
        self._initialize_wave_xy()

        self._rotate_coords()
        self._translate_coords()

    def _rotate_coords(self):
        """ Update all feature coordinates by rotating to the correct angle. """

        theta = np.arctan2(self.dy, self.dx)
        c = np.cos(theta)
        s = np.sin(theta)

        self.coords = {
            feature: (x * c - y * s, x * s + y * c)
            for feature, (x, y) in self.coords.items()
        }

    def _translate_coords(self):
        """ Update all feature coordinates by translating from (0,0) to the correct location. """

        self.coords = {
            feature: (self.x + feature_x, self.y + feature_y)
            for feature, (feature_x, feature_y) in self.coords.items()
        }

    def _initialize_heads_xy(self):
        """ Set the coordinates for all arrowheads.
        Coordinates are initialized as if the arrow was going straight to the right from (0, 0).
        """

        # Avoid overlapping heads for very short arrows.
        head_length = min(self.head_length, self.arrow_length / self.n_heads) if self.n_heads > 0 else 0
        half_width = self.head_width / 2

        if self.has_left_head:
            left_head_x = np.array([0, -head_length, 0])
            left_head_y = np.array([half_width, 0, -half_width])

            if self.length_includes_head:
                left_head_x += head_length
        else:
            left_head_x = np.array([])
            left_head_y = np.array([])

        if self.has_right_head:
            right_head_x = np.array([self.arrow_length, self.arrow_length + head_length, self.arrow_length])
            right_head_y = np.array([half_width, 0, -half_width])

            if self.length_includes_head:
                right_head_x -= head_length
        else:
            right_head_x = np.array([])
            right_head_y = np.array([])

        self.coords["left_head"] = (left_head_x, left_head_y)
        self.coords["right_head"] = (right_head_x, right_head_y)

    def _initialize_stems_xy(self):
        """ Set the coordinates for all arrow stems.
        The stem of the arrow is the line between the arrowhead and the wavy part of the shaft.
        Coordinates are initialized as if the arrow was going straight to the right from (0, 0).
        """

        # Starting x-coordinate of left stem for an arrow starting at (0,0).
        start_x = (
                self.head_length
                # If arrow doesn't include a left arrowhead, don't need a stem.
                * self.has_left_head
                # If the arrowhead length isn't included or the arrow is open, can start the shaft at (0,0).
                * self.length_includes_head
                * self.is_closed
        )

        # Ending x-coordinate of the right stem for an arrow starting at (0,0).
        # Similar to the starting x-coordinate of the left stem, but needs to be subtracted from the total length
        # of the arrow.
        end_x = (
            self.arrow_length
            - self.head_length * self.has_right_head * self.length_includes_head * self.is_closed
        )

        # Reduce length of stem(s) when arrowheads are overly close together.
        stem_length = (
            min(self.stem_length, (end_x - start_x) / self.n_heads)
            if self.n_heads > 0
            else 0
        )

        if self.has_left_head and stem_length > 0:
            left_stem_x = np.array([start_x, start_x + stem_length])
            left_stem_y = np.array([0, 0])
        else:
            left_stem_x = np.array([])
            left_stem_y = np.array([])

        if self.has_right_head and stem_length > 0:
            right_stem_x = np.array([end_x - stem_length, end_x])
            right_stem_y = np.array([0, 0])
        else:
            right_stem_x = np.array([])
            right_stem_y = np.array([])

        self.coords["left_stem"] = (left_stem_x, left_stem_y)
        self.coords["right_stem"] = (right_stem_x, right_stem_y)

    def _initialize_wave_xy(self):
        """ Set the coordinates for the wavy shaft of the arrow.
        Coordinates are initialized as if the arrow was going straight to the right from (0, 0).
        """

        # The length of the arrowhead and stem together.
        head_stem = self.length_includes_head * self.head_length + self.stem_length
        wave_length = self.arrow_length - head_stem * self.n_heads

        if wave_length < 0:
            self.coords["wave"] = np.array([]), np.array([])
        else:
            if self.n_waves is None:
                # Round to the nearest half-wave to end the wave at the middle of the arrowhead.
                self.n_waves = np.round(wave_length * self.wave_frequency * 2) / 2

            t = np.linspace(0, 1, self.resolution)
            wave_x = t * wave_length + head_stem * self.has_left_head    # Start at the end of the arrowhead and stem.
            wave_y = np.sin(t * 2 * np.pi * self.n_waves) * self.wave_height

            self.coords["wave"] = (wave_x, wave_y)


def plot_wavy_arrow(*args, **kwargs):
    """ Plot an arrow with a sine wave for a shaft.

    Accepts any parameters that can be passed to a call of WavyArrow.

    Parameters:
        x: float
            The x-coordinate of the base of the arrow.

        y: float
            The y-coordinate of the base of the arrow.

        dx: float
            The length of the arrow in the x direction.

        dy: float
            The length of the arrow in the y direction.

        n_waves: float (optional)
            The number of full sine waves in the arrow. If not provided, will be calculated using wave_frequency.

        wave_frequency: float (default=0.25)
            The number of full waves per 1 unit in plotting coordinates.
            Will be rounded to the nearest half-wave.
            Ignored if n_waves is not None.

        wave_height: float (default=1)
            The height of the crest of the wave.
            The full wave height (crest and trough) will be twice this value.

        resolution: int (default=500)
            The number of coordinates used in creating the wave.

        stem_length: float (default=1)
            The length of the stem(s) connecting the arrowhead(s) and the wave.
            The stem is used to connect the wave at the middle of the arrowhead.

        has_left_head: bool (default=False)
            Whether to include an arrowhead on the left side (x,y) of the arrow.

        has_right_head: bool (default=True)
            Whether to include an arrowhead on the right side (x + dx, y + dy) of the arrow.

        head_length: float (default=3)
            The length of the arrowhead (shaft to tip).

        head_width: float (default=4)
            The width of the base of the arrowhead.

        length_includes_head: bool (default=True)
            Whether the length of the arrow includes the arrowhead.

        is_closed: bool (default=True)
            Whether the arrowhead is closed or not.
                True: -|>
                False: ->

        shaft_kw: dict (optional)
            Additional keyword arguments to be provided to the .plot() function for the shaft of the arrow.

        head_kw: dict (optional)
            Additional keyword arguments to be provided to the .plot() or .fill() function for any arrowheads.

        left_head_kw: dict (optional)
            Additional keyword arguments to be provided to the .plot() or .fill() function specifically for the
            left arrowhead. Will supersede head_kw.

        right_head_kw: dict (optional)
            Additional keyword arguments to be provided to the .plot() or .fill() function specifically for the
            right arrowhead. Will supersede head_kw.

        ax: matplotlib Axes (optional)
            If not provided, will use the currently active Axes.

        kwargs: Any other properties that can be provided to the plotting functions for both the shaft and the
            arrowheads.

    Returns:
        WavyArrow
    """

    wavy_arrow = WavyArrow()
    return wavy_arrow(*args, **kwargs)
