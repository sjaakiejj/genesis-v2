"""
Elevation Map Generator

Generates colored elevation maps from plate and feature data.
Includes continental shelves, mountains, rift valleys, and volcanic features.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math
from typing import List, Optional, Tuple, Any
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from shapely.ops import unary_union
from scipy.ndimage import distance_transform_edt, gaussian_filter


class ElevationMapGenerator:
    """Generates colored elevation maps from plate and tectonic feature data."""

    # Elevation values (in meters, normalized later)
    DEEP_OCEAN = -4000
    MID_OCEAN = -2000
    SHALLOW_OCEAN = -500
    CONTINENTAL_SHELF = -200
    COAST = 0
    LOWLANDS = 200
    HIGHLANDS = 500
    MOUNTAINS = 1500
    HIGH_MOUNTAINS = 3000
    PEAKS = 4500

    # Color palette (RGB)
    COLORS = {
        -4000: (25, 50, 100),  # Deep ocean
        -2000: (40, 80, 140),  # Ocean
        -500: (80, 140, 180),  # Shallow ocean
        -200: (120, 180, 200),  # Continental shelf
        0: (210, 190, 150),  # Coast/Beach
        200: (230, 200, 140),  # Lowlands
        500: (220, 170, 100),  # Highlands
        1500: (200, 100, 50),  # Mountains
        3000: (150, 50, 30),  # High mountains
        4500: (120, 30, 20),  # Peaks
    }

    def __init__(self, width: int = 2048, height: int = 1024):
        self.width = width
        self.height = height
        self._elevation = None  # Will store elevation values as numpy array

    def generate(self, plates: List[Any]) -> Image.Image:
        """
        Generate a colored elevation map from plate data.

        Args:
            plates: List of PlateData objects with features

        Returns:
            PIL Image with colored elevation map
        """
        # Initialize elevation array (all deep ocean)
        self._elevation = np.full(
            (self.height, self.width), self.DEEP_OCEAN, dtype=np.float32
        )

        # Build masks for different terrain types
        land_mask = self._create_land_mask(plates)
        continent_mask = self._create_continent_mask(plates)

        # 1. Set base elevations
        self._paint_base_elevations(land_mask, continent_mask)

        # 2. Add continental shelves
        self._paint_continental_shelf(continent_mask)

        # 3. Add mountains at collision zones
        self._paint_mountains(plates)

        # 4. Add rift valleys
        self._paint_rift_valleys(plates)

        # 5. Add volcanic features
        self._paint_volcanic_features(plates)

        # 6. Add mid-ocean ridges
        self._paint_ridges(plates)

        # 7. Apply coastal noise
        self._apply_coastal_noise(continent_mask)

        # 8. Apply smoothing
        self._apply_smoothing()

        # 9. Convert elevation to color
        return self._elevation_to_color()

    def _lon_lat_to_pixel(self, lon: float, lat: float) -> Tuple[int, int]:
        """Convert longitude/latitude to pixel coordinates."""
        x = int((lon + 180) / 360 * self.width) % self.width
        y = int((90 - lat) / 180 * self.height)
        y = max(0, min(self.height - 1, y))
        return x, y

    def _create_land_mask(self, plates: List[Any]) -> np.ndarray:
        """Create a binary mask of all plate polygons."""
        mask = np.zeros((self.height, self.width), dtype=bool)
        img = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(img)

        for plate in plates:
            if plate.boundary_polygon:
                self._draw_polygon_to_mask(draw, plate.boundary_polygon)

        mask = np.array(img) > 0
        return mask

    def _create_continent_mask(self, plates: List[Any]) -> np.ndarray:
        """Create a binary mask of continent polygons."""
        mask = np.zeros((self.height, self.width), dtype=bool)
        img = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(img)

        for plate in plates:
            if plate.continent_polygon:
                self._draw_polygon_to_mask(draw, plate.continent_polygon)

        mask = np.array(img) > 0
        return mask

    def _draw_polygon_to_mask(self, draw: ImageDraw.Draw, geom):
        """Draw a polygon geometry to an image mask."""
        if hasattr(geom, "geom_type"):
            if geom.geom_type == "Polygon":
                coords = list(geom.exterior.coords)
                pixels = [self._lon_lat_to_pixel(lon, lat) for lon, lat in coords]
                draw.polygon(pixels, fill=255)
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    coords = list(poly.exterior.coords)
                    pixels = [self._lon_lat_to_pixel(lon, lat) for lon, lat in coords]
                    draw.polygon(pixels, fill=255)

    def _paint_base_elevations(self, land_mask: np.ndarray, continent_mask: np.ndarray):
        """Set base elevation values for ocean and land."""
        # Ocean stays at deep ocean (already set)

        # Continental crust plates get shallow ocean
        self._elevation[land_mask & ~continent_mask] = self.SHALLOW_OCEAN

        # Actual continents get lowland elevation
        self._elevation[continent_mask] = self.LOWLANDS

    def _paint_continental_shelf(self, continent_mask: np.ndarray):
        """Paint continental shelf as buffer around landmasses."""
        # Create distance transform from continent edges
        distance = distance_transform_edt(~continent_mask)

        # Shelf width in pixels (roughly 5 degrees or so)
        shelf_width = int(self.width * 0.02)  # ~2% of map width

        # Paint shelf where close to continent but still in ocean
        shelf_zone = (distance > 0) & (distance < shelf_width)

        # Gradient from coast to deep
        for y in range(self.height):
            for x in range(self.width):
                if shelf_zone[y, x]:
                    dist = distance[y, x]
                    ratio = dist / shelf_width
                    # Interpolate from COAST to SHALLOW_OCEAN
                    elev = self.COAST * (1 - ratio) + self.CONTINENTAL_SHELF * ratio
                    self._elevation[y, x] = max(self._elevation[y, x], elev)

    def _paint_mountains(self, plates: List[Any]):
        """Paint mountains at collision zones with stepped gradient."""
        for plate in plates:
            if not hasattr(plate, "features"):
                continue

            for feature in plate.features:
                if feature.feature_type == "mountain_range" and feature.line:
                    self._paint_mountain_line(feature.line)
                elif feature.feature_type == "mountain_plateau" and feature.polygon:
                    self._paint_mountain_polygon(feature.polygon)

    def _paint_mountain_line(self, geom):
        """Paint mountains along a boundary line with stepped gradient."""
        # Create a temporary image to draw the mountain influence
        influence_img = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(influence_img)

        # Draw thick line for mountain core
        if hasattr(geom, "geom_type"):
            if geom.geom_type == "LineString":
                coords = list(geom.coords)
                pixels = [self._lon_lat_to_pixel(lon, lat) for lon, lat in coords]
                if len(pixels) >= 2:
                    draw.line(pixels, fill=255, width=8)
            elif geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    coords = list(line.coords)
                    pixels = [self._lon_lat_to_pixel(lon, lat) for lon, lat in coords]
                    if len(pixels) >= 2:
                        draw.line(pixels, fill=255, width=8)

        # Apply blur for gradient
        influence = np.array(influence_img.filter(ImageFilter.GaussianBlur(radius=20)))

        # Paint elevation based on influence (stepped)
        steps = [
            (200, self.PEAKS),
            (150, self.HIGH_MOUNTAINS),
            (100, self.MOUNTAINS),
            (50, self.HIGHLANDS),
            (20, self.HIGHLANDS - 100),
        ]

        for threshold, elevation in steps:
            mask = influence > threshold
            # Add some randomness
            noise = np.random.randn(self.height, self.width) * 50
            self._elevation[mask] = np.maximum(
                self._elevation[mask], elevation + noise[mask]
            )

    def _paint_mountain_polygon(self, geom):
        """Paint mountain plateau polygon with elevation."""
        img = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(img)
        self._draw_polygon_to_mask(draw, geom)

        # Apply blur for gradient edges
        blurred = np.array(img.filter(ImageFilter.GaussianBlur(radius=15)))

        # Set elevation based on intensity
        steps = [
            (200, self.HIGH_MOUNTAINS),
            (100, self.MOUNTAINS),
            (30, self.HIGHLANDS),
        ]

        for threshold, elevation in steps:
            mask = blurred > threshold
            noise = np.random.randn(self.height, self.width) * 30
            self._elevation[mask] = np.maximum(
                self._elevation[mask], elevation + noise[mask]
            )

    def _paint_rift_valleys(self, plates: List[Any]):
        """Paint rift valleys at divergent boundaries on continents."""
        for plate in plates:
            if not hasattr(plate, "boundary_segments"):
                continue

            from tectonic_engine import BoundaryType, CrustType

            for seg in plate.boundary_segments:
                if seg.boundary_type != BoundaryType.DIVERGENT:
                    continue

                # Check if on continental crust
                if plate.crust_type != CrustType.CONTINENTAL:
                    continue

                # Draw rift valley
                if seg.geometry:
                    self._paint_valley_line(seg.geometry)

    def _paint_valley_line(self, geom):
        """Paint a rift valley as a depression along a line."""
        influence_img = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(influence_img)

        if hasattr(geom, "geom_type"):
            if geom.geom_type == "LineString":
                coords = list(geom.coords)
                pixels = [self._lon_lat_to_pixel(lon, lat) for lon, lat in coords]
                if len(pixels) >= 2:
                    draw.line(pixels, fill=255, width=4)
            elif geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    coords = list(line.coords)
                    pixels = [self._lon_lat_to_pixel(lon, lat) for lon, lat in coords]
                    if len(pixels) >= 2:
                        draw.line(pixels, fill=255, width=4)

        influence = np.array(influence_img.filter(ImageFilter.GaussianBlur(radius=8)))

        # Lower elevation where influence is high
        mask = influence > 50
        self._elevation[mask] = np.minimum(
            self._elevation[mask], self.COAST - 100  # Slight depression
        )

    def _paint_volcanic_features(self, plates: List[Any]):
        """Paint volcanic arcs, islands, and calderas."""
        for plate in plates:
            if not hasattr(plate, "features"):
                continue

            for feature in plate.features:
                if feature.feature_type == "volcanic_arc" and feature.line:
                    self._paint_volcanic_arc(feature.line)
                elif feature.feature_type == "hotspot" and feature.location is not None:
                    self._paint_hotspot(feature.location, plate)

    def _paint_volcanic_arc(self, geom):
        """Paint volcanic arc as a chain of elevated points."""
        if not hasattr(geom, "geom_type"):
            return

        coords_list = []
        if geom.geom_type == "LineString":
            coords_list = [list(geom.coords)]
        elif geom.geom_type == "MultiLineString":
            coords_list = [list(line.coords) for line in geom.geoms]

        for coords in coords_list:
            # Sample points along the arc for volcano placement
            for i in range(0, len(coords), max(1, len(coords) // 8)):
                lon, lat = coords[i]
                self._paint_volcano(lon, lat)

    def _paint_volcano(self, lon: float, lat: float, radius_px: int = 6):
        """Paint a single volcano as an elevated cone."""
        cx, cy = self._lon_lat_to_pixel(lon, lat)

        for dy in range(-radius_px, radius_px + 1):
            for dx in range(-radius_px, radius_px + 1):
                x = (cx + dx) % self.width
                y = cy + dy
                if 0 <= y < self.height:
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist < radius_px:
                        # Conical elevation profile
                        elev = self.MOUNTAINS + (1 - dist / radius_px) * 500
                        self._elevation[y, x] = max(self._elevation[y, x], elev)

    def _paint_hotspot(self, location: np.ndarray, plate: Any):
        """Paint hotspot as island (ocean) or caldera (land)."""
        x, y, z = location
        r = math.sqrt(x**2 + y**2 + z**2)
        lat = math.degrees(math.asin(z / r))
        lon = math.degrees(math.atan2(y, x))

        px, py = self._lon_lat_to_pixel(lon, lat)

        # Check if on land or ocean
        is_on_land = self._elevation[py, px] > self.COAST

        if is_on_land:
            # Caldera - depression with raised rim
            self._paint_caldera(px, py)
        else:
            # Island chain
            self._paint_island_chain(px, py)

    def _paint_caldera(self, cx: int, cy: int, radius: int = 8):
        """Paint a caldera (volcanic crater on land)."""
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x = (cx + dx) % self.width
                y = cy + dy
                if 0 <= y < self.height:
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist < radius * 0.6:
                        # Inner crater - depression
                        self._elevation[y, x] = max(
                            self.LOWLANDS, self._elevation[y, x] - 200
                        )
                    elif dist < radius:
                        # Raised rim
                        self._elevation[y, x] = max(
                            self._elevation[y, x], self.HIGHLANDS + 200
                        )

    def _paint_island_chain(self, cx: int, cy: int):
        """Paint a chain of volcanic islands in the ocean."""
        # Create 3-5 islands in a rough line
        num_islands = np.random.randint(3, 6)
        angle = np.random.random() * 2 * math.pi

        for i in range(num_islands):
            offset = i * 8 - (num_islands * 4)
            ix = int(cx + offset * math.cos(angle)) % self.width
            iy = int(cy + offset * math.sin(angle))
            iy = max(0, min(self.height - 1, iy))

            # Paint small island
            radius = np.random.randint(2, 5)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    x = (ix + dx) % self.width
                    y = iy + dy
                    if 0 <= y < self.height:
                        dist = math.sqrt(dx**2 + dy**2)
                        if dist < radius:
                            elev = self.LOWLANDS + (1 - dist / radius) * 300
                            self._elevation[y, x] = max(self._elevation[y, x], elev)

    def _paint_ridges(self, plates: List[Any]):
        """Paint mid-ocean ridges as elevated seafloor."""
        for plate in plates:
            if not hasattr(plate, "features"):
                continue

            for feature in plate.features:
                if feature.feature_type == "ridge" and feature.line:
                    self._paint_ridge_line(feature.line)

    def _paint_ridge_line(self, geom):
        """Paint mid-ocean ridge as slightly elevated seafloor."""
        influence_img = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(influence_img)

        if hasattr(geom, "geom_type"):
            if geom.geom_type == "LineString":
                coords = list(geom.coords)
                pixels = [self._lon_lat_to_pixel(lon, lat) for lon, lat in coords]
                if len(pixels) >= 2:
                    draw.line(pixels, fill=255, width=12)
            elif geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    coords = list(line.coords)
                    pixels = [self._lon_lat_to_pixel(lon, lat) for lon, lat in coords]
                    if len(pixels) >= 2:
                        draw.line(pixels, fill=255, width=12)

        influence = np.array(influence_img.filter(ImageFilter.GaussianBlur(radius=15)))

        # Elevate ocean floor at ridges
        mask = (influence > 30) & (self._elevation < self.COAST)
        ratio = influence[mask] / 255.0
        self._elevation[mask] = self.MID_OCEAN + ratio * 1000  # Raised seafloor

    def _apply_coastal_noise(self, continent_mask: np.ndarray):
        """Apply noise to coastal boundaries for more natural look."""
        # Create distance from coast
        distance = distance_transform_edt(continent_mask)
        coast_zone = (distance > 0) & (distance < 30)

        # Generate noise
        noise = np.random.randn(self.height, self.width) * 100
        noise = gaussian_filter(noise, sigma=3)

        # Apply noise to coast zone
        self._elevation[coast_zone] += noise[coast_zone]

    def _apply_smoothing(self):
        """Apply light smoothing to reduce harsh transitions."""
        self._elevation = gaussian_filter(self._elevation, sigma=1.5)

    def _elevation_to_color(self) -> Image.Image:
        """Convert elevation array to colored image."""
        # Create RGB image
        img = Image.new("RGB", (self.width, self.height))
        pixels = img.load()

        # Get sorted elevation thresholds
        thresholds = sorted(self.COLORS.keys())

        for y in range(self.height):
            for x in range(self.width):
                elev = self._elevation[y, x]
                color = self._interpolate_color(elev, thresholds)
                pixels[x, y] = color

        return img

    def _interpolate_color(
        self, elev: float, thresholds: List[int]
    ) -> Tuple[int, int, int]:
        """Interpolate between color stops based on elevation."""
        # Find bracketing thresholds
        lower_t = thresholds[0]
        upper_t = thresholds[-1]

        for i, t in enumerate(thresholds):
            if elev < t:
                if i > 0:
                    lower_t = thresholds[i - 1]
                    upper_t = t
                break
            lower_t = t
            if i < len(thresholds) - 1:
                upper_t = thresholds[i + 1]

        # Get colors
        color_low = self.COLORS.get(lower_t, (100, 100, 100))
        color_high = self.COLORS.get(upper_t, (100, 100, 100))

        # Interpolate
        if upper_t == lower_t:
            return color_low

        ratio = (elev - lower_t) / (upper_t - lower_t)
        ratio = max(0, min(1, ratio))

        r = int(color_low[0] + (color_high[0] - color_low[0]) * ratio)
        g = int(color_low[1] + (color_high[1] - color_low[1]) * ratio)
        b = int(color_low[2] + (color_high[2] - color_low[2]) * ratio)

        return (r, g, b)

    def get_elevation_array(self) -> np.ndarray:
        """Return the raw elevation array (for export or further processing)."""
        return self._elevation.copy() if self._elevation is not None else None
