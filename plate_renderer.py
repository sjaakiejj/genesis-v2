"""
Plate Renderer - Panda3D Visualization Module
Renders plates as a texture applied to a procedural sphere.
Supports threaded generation with progress callbacks and GPU-based selection highlighting.
"""

from panda3d.core import (
    GeomVertexFormat,
    GeomVertexData,
    GeomVertexWriter,
    Geom,
    GeomTriangles,
    GeomNode,
    NodePath,
    Texture,
    PNMImage,
    LColor,
    Shader,
    TextureStage,
    LineSegs,
)
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Callable, Optional, Set, TYPE_CHECKING, Tuple
import math
import threading
from queue import Queue
from dataclasses import dataclass

if TYPE_CHECKING:
    from tectonic_engine import PlateData


@dataclass
class GenerationProgress:
    """Progress update from texture generation."""

    current: int
    total: int
    percentage: float
    message: str


class PlateTextureGenerator:
    """
    Generates equirectangular plate textures using Pillow.
    Outputs both a color texture and a plate ID map.
    """

    def __init__(self, width: int = 1024, height: int = 512):
        """
        Initialize texture generator.

        Args:
            width: Texture width in pixels
            height: Texture height in pixels
        """
        self.width = width
        self.height = height
        self._cancel_flag = False

    def generate_texture(
        self,
        plates: List["PlateData"],
        progress_callback: Optional[Callable[[GenerationProgress], None]] = None,
        show_plates: bool = True,
    ) -> Optional[Tuple[Image.Image, Image.Image]]:
        """
        Generate equirectangular plate texture and ID map.

        Args:
            plates: List of PlateData with vertices and colors
            progress_callback: Optional callback for progress updates
            show_plates: If True, show plate colors/borders. If False, just ocean+continents.

        Returns:
            Tuple of (color_image, id_image), or None if cancelled.
            id_image encodes plate ID in red channel.
        """
        self._cancel_flag = False

        # Create RGB image for color
        color_image = Image.new("RGB", (self.width, self.height), (30, 60, 90))
        color_pixels = color_image.load()

        # Create grayscale image for IDs (0-255)
        # Using "L" mode (8-bit pixels, black and white)
        id_image = Image.new("L", (self.width, self.height), 255)  # 255 = no plate
        id_pixels = id_image.load()

        # Build plate ID map for internal mapping
        # Warning: if we have > 255 plates, this simple 8-bit map won't work well
        # But for 12-50 plates it is fine.

        # Use vector drawing instead of pixel iteration
        self._draw_plate_polygons(color_image, id_image, plates, show_plates)

        if progress_callback:
            progress = GenerationProgress(
                current=self.height,
                total=self.height,
                percentage=100.0,
                message="Texture generation complete",
            )
            progress_callback(progress)

        return color_image, id_image

    def _draw_plate_polygons(
        self,
        color_img: Image.Image,
        id_img: Image.Image,
        plates: List["PlateData"],
        show_plates: bool = True,
    ):
        """Draw plate polygons onto the textures.

        Args:
            color_img: Color texture to draw on
            id_img: ID map texture
            plates: List of plates to draw
            show_plates: If True, draw plate colors/borders. If False, only ocean+continents.
        """
        draw_color = ImageDraw.Draw(color_img)
        draw_id = ImageDraw.Draw(id_img)

        # Generate unique color map for IDs if needed, or simple grayscale
        # We assume IDs fit in 0-255 for now

        plate_id_to_index = {plate.plate_id: i for i, plate in enumerate(plates)}

        # Helper to map lat/lon to pixel coords
        def to_pixels(lon, lat):
            # lon: -180 to 180 -> 0 to width
            x = (lon + 180.0) / 360.0 * self.width
            # lat: 90 to -90 -> 0 to height (flipped Y)
            y = (90.0 - lat) / 180.0 * self.height
            return x, y

        # Ocean color for simple mode
        ocean_color = (30, 60, 90)  # Dark blue ocean

        # First pass: Draw plate backgrounds (or just ocean in simple mode)
        for plate in plates:
            if not plate.boundary_polygon:
                continue

            if show_plates:
                # Color based on CrustType
                fill_color = (
                    int(plate.color[0] * 255),
                    int(plate.color[1] * 255),
                    int(plate.color[2] * 255),
                )

                if hasattr(plate, "crust_type") and plate.crust_type:
                    if plate.crust_type.name == "OCEANIC":
                        fill_color = (135, 206, 235)  # Light Blue (Sky Blue)
                    elif plate.crust_type.name == "CONTINENTAL":
                        fill_color = (144, 238, 144)  # Light Green (Pale Green)
            else:
                # Simple mode: everything is ocean blue (continents drawn on top)
                fill_color = ocean_color

            # ID index
            idx = plate_id_to_index.get(plate.plate_id, 255)

            # Draw each polygon in the MultiPolygon
            for poly in plate.boundary_polygon.geoms:
                # Get exterior coordinates
                pixels = [to_pixels(*coord) for coord in poly.exterior.coords]

                # Draw filled polygon
                draw_color.polygon(pixels, fill=fill_color)
                draw_id.polygon(pixels, fill=idx)

        # Second pass: Draw continents on top of continental plates
        continent_fill = (210, 180, 140)  # Tan/Brown for land
        coastline_color = (
            (101, 67, 33) if show_plates else None
        )  # Only show coastline in plates mode

        for plate in plates:
            if not hasattr(plate, "continent_polygon") or not plate.continent_polygon:
                continue

            # Draw continent polygons
            for poly in plate.continent_polygon.geoms:
                pixels = [to_pixels(*coord) for coord in poly.exterior.coords]

                # Draw filled continent
                draw_color.polygon(pixels, fill=continent_fill)

                # Draw coastline border only in plates mode
                if coastline_color:
                    draw_color.polygon(pixels, outline=coastline_color)

        # Third pass: Draw tectonic features
        self._draw_features(draw_color, plates, to_pixels)

    def _draw_features(self, draw, plates, to_pixels):
        """Draw tectonic features like volcanic arcs, hotspots, mountains, ridges."""
        # Feature colors
        volcanic_arc_color = (255, 69, 0)  # Red-Orange for volcanic arcs
        hotspot_color = (255, 0, 0)  # Red for hotspots
        mountain_color = (139, 90, 43)  # Saddle brown for mountains
        ridge_color = (255, 140, 0)  # Dark orange for ridges

        for plate in plates:
            if not hasattr(plate, "features") or not plate.features:
                continue

            for feature in plate.features:
                if feature.feature_type == "volcanic_arc" and feature.line:
                    # Draw volcanic arc as line of triangles
                    self._draw_line_feature(
                        draw, feature.line, to_pixels, volcanic_arc_color, width=2
                    )
                    # Add triangle markers along the line
                    self._draw_volcano_markers(
                        draw, feature.line, to_pixels, volcanic_arc_color
                    )

                elif feature.feature_type == "hotspot" and feature.location is not None:
                    # Draw hotspot as a red dot
                    # Convert 3D to lat/lon
                    x, y, z = feature.location
                    r = (x**2 + y**2 + z**2) ** 0.5
                    lat = math.degrees(math.asin(z / r))
                    lon = math.degrees(math.atan2(y, x))
                    px, py = to_pixels(lon, lat)
                    # Draw filled circle
                    radius = 4
                    draw.ellipse(
                        [px - radius, py - radius, px + radius, py + radius],
                        fill=hotspot_color,
                        outline=(200, 0, 0),
                    )

                elif feature.feature_type == "mountain_range" and feature.line:
                    # Draw mountain range as thick brown line
                    self._draw_line_feature(
                        draw, feature.line, to_pixels, mountain_color, width=4
                    )

                elif feature.feature_type == "ridge" and feature.line:
                    # Draw mid-ocean ridge as orange line
                    self._draw_line_feature(
                        draw, feature.line, to_pixels, ridge_color, width=2
                    )

                elif feature.feature_type == "mountain_plateau" and feature.polygon:
                    # Draw mountain plateau as dark gray/brown filled polygon
                    plateau_color = (120, 100, 80)  # Dark grayish brown
                    self._draw_polygon_feature(
                        draw, feature.polygon, to_pixels, plateau_color
                    )

    def _draw_line_feature(self, draw, geom, to_pixels, color, width=2):
        """Draw a line or multiline feature."""
        if hasattr(geom, "geom_type"):
            if geom.geom_type == "LineString":
                coords = list(geom.coords)
                if len(coords) >= 2:
                    pixels = [to_pixels(lon, lat) for lon, lat in coords]
                    draw.line(pixels, fill=color, width=width)
            elif geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    coords = list(line.coords)
                    if len(coords) >= 2:
                        pixels = [to_pixels(lon, lat) for lon, lat in coords]
                        draw.line(pixels, fill=color, width=width)

    def _draw_volcano_markers(self, draw, geom, to_pixels, color, spacing=30):
        """Draw small triangle markers along a line to represent volcanoes."""
        import math

        def draw_triangle(cx, cy, size=5):
            """Draw a small triangle at (cx, cy)."""
            points = [
                (cx, cy - size),  # Top
                (cx - size, cy + size),  # Bottom left
                (cx + size, cy + size),  # Bottom right
            ]
            draw.polygon(points, fill=color)

        if hasattr(geom, "geom_type"):
            if geom.geom_type == "LineString":
                coords = list(geom.coords)
                # Draw markers at intervals
                for i in range(0, len(coords), max(1, len(coords) // 5)):
                    lon, lat = coords[i]
                    px, py = to_pixels(lon, lat)
                    draw_triangle(px, py)
            elif geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    coords = list(line.coords)
                    for i in range(0, len(coords), max(1, len(coords) // 3)):
                        lon, lat = coords[i]
                        px, py = to_pixels(lon, lat)
                        draw_triangle(px, py)

    def _draw_polygon_feature(self, draw, geom, to_pixels, color):
        """Draw a filled polygon feature (for plateaus, etc)."""
        if hasattr(geom, "geom_type"):
            if geom.geom_type == "Polygon":
                coords = list(geom.exterior.coords)
                if len(coords) >= 3:
                    pixels = [to_pixels(lon, lat) for lon, lat in coords]
                    draw.polygon(pixels, fill=color, outline=(80, 60, 40))
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    coords = list(poly.exterior.coords)
                    if len(coords) >= 3:
                        pixels = [to_pixels(lon, lat) for lon, lat in coords]
                        draw.polygon(pixels, fill=color, outline=(80, 60, 40))

    def cancel(self):
        """Cancel ongoing generation."""
        self._cancel_flag = True


class ThreadedTextureGenerator:
    """
    Manages threaded texture generation with progress updates.
    """

    def __init__(self):
        self.generator = PlateTextureGenerator()
        self._thread: Optional[threading.Thread] = None
        self._result_queue: Queue = Queue()
        self._progress_queue: Queue = Queue()
        self._is_generating = False

    def start_generation(self, plates: List["PlateData"], show_plates: bool = True):
        """
        Start texture generation in a background thread.

        Args:
            plates: List of PlateData to generate texture from
            show_plates: If True, show plate colors/borders. If False, just ocean+continents.
        """
        if self._is_generating:
            self.cancel()

        self._is_generating = True
        self._thread = threading.Thread(
            target=self._generation_worker, args=(plates, show_plates), daemon=True
        )
        self._thread.start()

    def _generation_worker(self, plates: List["PlateData"], show_plates: bool):
        """Worker thread for texture generation."""
        try:
            result = self.generator.generate_texture(
                plates, progress_callback=self._on_progress, show_plates=show_plates
            )
            # result is tuple (color_image, id_image)
            self._result_queue.put(("success", result))
        except Exception as e:
            self._result_queue.put(("error", str(e)))
        finally:
            # Only reset if we are still the active thread
            if self._thread == threading.current_thread():
                self._is_generating = False

    def _on_progress(self, progress: GenerationProgress):
        """Handle progress updates from generator."""
        self._progress_queue.put(progress)

    def get_progress(self) -> Optional[GenerationProgress]:
        """Get latest progress update (non-blocking)."""
        latest = None
        while not self._progress_queue.empty():
            try:
                latest = self._progress_queue.get_nowait()
            except:
                break
        return latest

    def get_result(self) -> Optional[tuple]:
        """Get generation result if available (non-blocking)."""
        if not self._result_queue.empty():
            try:
                return self._result_queue.get_nowait()
            except:
                pass
        return None

    def is_generating(self) -> bool:
        """Check if generation is in progress."""
        return self._is_generating

    def cancel(self):
        """Cancel ongoing generation."""
        self.generator.cancel()
        self._is_generating = False


class PlateRenderer:
    """
    Renders tectonic plates using a textured sphere with custom shaders.
    """

    def __init__(self, parent_node: NodePath, scale: float = 2.0):
        """
        Initialize the plate renderer.
        """
        self.parent_node = parent_node
        self.scale = scale
        self.globe: NodePath = None
        self.threaded_generator = ThreadedTextureGenerator()

        self._current_color_texture: Optional[Image.Image] = None
        self._current_id_texture: Optional[Image.Image] = None
        self._reference_texture: Optional[Image.Image] = None

        self._plates: List["PlateData"] = []
        self._selected_ids: Set[int] = set()

        # Shader Inputs
        self._selection_tex: Texture = None
        self._vector_tex: Texture = None

    def setup_globe(self):
        """Create the globe sphere and load shaders."""
        if self.globe is not None:
            self.globe.removeNode()

        self.globe = self._create_uv_sphere(segments=64, rings=48)
        self.globe.setScale(self.scale)
        self.globe.reparentTo(self.parent_node)

        # Set a default ocean blue color
        self.globe.setColor(LColor(0.12, 0.24, 0.35, 1.0))

        # Load shaders
        try:
            shader = Shader.load(
                Shader.SL_GLSL,
                vertex="shaders/globe.vert",
                fragment="shaders/globe.frag",
            )
            self.globe.setShader(shader)
        except:
            print("Failed to load shaders! Using default rendering.")
            self.globe.clearShader()

        # Initialize selection texture (2D, 256x1 pixels) to satisfy Mac/Driver
        self._selection_tex = Texture("selection_texture")
        self._selection_tex.setup2dTexture(
            256, 1, Texture.T_unsigned_byte, Texture.F_red
        )
        self._selection_tex.setWrapU(Texture.WMClamp)
        self._selection_tex.setWrapV(Texture.WMClamp)
        self._selection_tex.setMagfilter(Texture.FTNearest)
        self._selection_tex.setMinfilter(Texture.FTNearest)

        # Initialize vector texture (transparent)
        self._vector_tex = Texture("vector_texture")
        self._vector_tex.setup2dTexture(
            1024, 512, Texture.T_unsigned_byte, Texture.F_rgba
        )
        self._vector_tex.setWrapU(Texture.WMRepeat)
        self._vector_tex.setWrapV(Texture.WMClamp)

        # Initialize with transparent
        img = PNMImage(256, 1, 1)
        img.fill(0)
        self._selection_tex.load(img)

        # Clear vector tex
        img_v = PNMImage(1024, 512, 4)
        img_v.fill(0, 0, 0)
        img_v.alphaFill(0)
        self._vector_tex.load(img_v)

        # Initialize elevation texture
        self._elevation_tex = Texture("elevation_texture")
        self._elevation_tex.setup2dTexture(
            2048, 1024, Texture.T_unsigned_byte, Texture.F_rgb
        )
        self._elevation_tex.setWrapU(Texture.WMRepeat)
        self._elevation_tex.setWrapV(Texture.WMClamp)
        self._elevation_visible = False
        self._has_elevation = False

        # Bind textures to shader input
        self.globe.setShaderInput("selection_tex", self._selection_tex)
        self.globe.setShaderInput("vector_tex", self._vector_tex)
        self.globe.setShaderInput("show_vectors", 1.0)
        self.globe.setShaderInput("show_borders", 1.0)

    def clear(self):
        """Remove existing globe geometry."""
        if self.globe is not None:
            self.globe.removeNode()
            self.globe = None

    def start_plate_generation(
        self,
        plates: List["PlateData"],
        selected_ids: Optional[Set[int]] = None,
        show_plates: bool = True,
    ):
        """
        Start async plate texture generation.

        Args:
            plates: List of plates to render
            selected_ids: Set of selected plate IDs for highlighting
            show_plates: If True, show plate colors/borders. If False, just ocean+continents.
        """
        self._plates = plates
        self._selected_ids = selected_ids or set()
        self._show_plates = show_plates
        # Update selection texture immediately to match new plate list/indices
        self._update_selection_texture()
        self.threaded_generator.start_generation(plates, show_plates)
        print(f"Started texture generation for {len(plates)} plates...")

    def refresh_selection(self, selected_ids: Set[int]):
        """
        Update selection state using 1D texture update (on GPU).
        Fast - no texture regeneration needed!
        """
        self._selected_ids = selected_ids
        self._update_selection_texture()

    def _update_selection_texture(self):
        """Update the 1D selection texture based on selected_ids."""
        if not self._plates:
            return

        # Map current plates to indices 0..N
        # We need to rely on the same order as generate_texture used
        # self._plates is the list passed.

        img = PNMImage(256, 1, 1)
        img.fill(0)

        for i, plate in enumerate(self._plates):
            if i >= 256:
                break  # Limit 256 plates for this optimization

            if plate.plate_id in self._selected_ids:
                img.setXel(i, 0, 1.0)  # Set to White (Selected)

        self._selection_tex.load(img)

    def update(self) -> Optional[GenerationProgress]:
        """
        Update method to be called each frame.
        """
        progress = self.threaded_generator.get_progress()

        result = self.threaded_generator.get_result()
        if result is not None:
            status, data = result
            if status == "success" and data is not None:
                color_img, id_img = data
                self._current_color_texture = color_img
                self._current_id_texture = id_img
                self._apply_textures(color_img, id_img)
                # Also apply current selection
                self._update_selection_texture()
                print("Textures applied to globe")
            elif status == "error":
                print(f"Texture generation error: {data}")

        return progress

    def is_generating(self) -> bool:
        """Check if texture generation is in progress."""
        return self.threaded_generator.is_generating()

    def get_current_texture(self) -> Optional[Image.Image]:
        """Get the current plate texture image (color only)."""
        return self._current_color_texture

    def get_plates(self) -> List["PlateData"]:
        """Get the current plates list."""
        return self._plates

    def generate_reference_texture(self, plates: List["PlateData"]) -> Image.Image:
        """
        Generate a reference Voronoi texture using pixel-based nearest neighbor.
        This provides a 'ground truth' visualization for debugging vectorization.
        """
        width, height = 512, 256  # Smaller size for debug view
        img = Image.new("RGB", (width, height), (0, 0, 0))
        pixels = img.load()

        # Precompute centroids (using seeed points for correct Voronoi matching)
        centroids = []
        for plate in plates:
            if plate.seed_point is not None:
                c = plate.seed_point / np.linalg.norm(plate.seed_point)
            elif plate.centroid is not None:
                c = plate.centroid / np.linalg.norm(plate.centroid)
            else:
                c = np.mean(plate.vertices, axis=0)
                c = c / np.linalg.norm(c)
            centroids.append((plate, c))

        # Scan pixels
        for y in range(height):
            lat = 90.0 - (y / height) * 180.0
            lat_rad = math.radians(lat)
            cos_lat = math.cos(lat_rad)
            sin_lat = math.sin(lat_rad)

            for x in range(width):
                lon = (x / width) * 360.0 - 180.0
                lon_rad = math.radians(lon)

                px = cos_lat * math.cos(lon_rad)
                py = cos_lat * math.sin(lon_rad)
                pz = sin_lat
                point = np.array([px, py, pz])

                closest = self._find_closest_plate(point, centroids)
                if closest:
                    r, g, b = closest.color
                    pixels[x, y] = (int(r * 255), int(g * 255), int(b * 255))

        self._reference_texture = img
        return img

    def get_reference_texture(self) -> Optional[Image.Image]:
        """Get the last generated reference texture."""
        return self._reference_texture

    def _find_closest_plate(self, point: np.ndarray, centroids: list) -> "PlateData":
        """Find the plate whose centroid is closest to the given point."""
        closest = None
        min_dist = float("inf")

        for plate, centroid in centroids:
            dot = np.dot(point, centroid)
            dist = -dot

            if dist < min_dist:
                min_dist = dist
                closest = plate

        return closest

    def _apply_textures(self, color_img: Image.Image, id_img: Image.Image):
        """Apply PIL images as textures to the globe stages."""
        if self.globe is None:
            self.setup_globe()

        # 1. Color Texture (Stage 0, implicit "tex")
        tex_color = self._pil_to_texture(color_img, "plate_color")
        self.globe.setTexture(tex_color)

        # 2. ID Texture (Custom Stage)
        tex_id = self._pil_to_texture(id_img, "plate_id", is_grayscale=True)
        tex_id.setMagfilter(Texture.FTNearest)  # Important: No filtering for IDs
        tex_id.setMinfilter(Texture.FTNearest)

        # We assign to a shader input
        self.globe.setShaderInput("id_tex", tex_id)

        self.globe.clearColor()

    def _create_uv_sphere(self, segments: int = 64, rings: int = 48) -> NodePath:
        """Create a UV-mapped sphere for texture mapping."""
        format = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData("sphere", format, Geom.UHStatic)

        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        texcoord = GeomVertexWriter(vdata, "texcoord")

        radius = 1.0

        for ring in range(rings + 1):
            phi = math.pi * ring / rings
            v = ring / rings

            for seg in range(segments + 1):
                # Add pi to rotate 180 degrees to match texture projection
                theta = 2.0 * math.pi * seg / segments + math.pi
                u = seg / segments

                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.sin(phi) * math.sin(theta)
                z = radius * math.cos(phi)

                vertex.addData3(x, y, z)
                normal.addData3(x, y, z)
                texcoord.addData2(u, 1.0 - v)

        prim = GeomTriangles(Geom.UHStatic)

        for ring in range(rings):
            for seg in range(segments):
                i0 = ring * (segments + 1) + seg
                i1 = i0 + 1
                i2 = i0 + (segments + 1)
                i3 = i2 + 1

                prim.addVertices(i0, i2, i1)
                prim.addVertices(i1, i2, i3)

        prim.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)

        node = GeomNode("globe_sphere")
        node.addGeom(geom)

        return NodePath(node)

    def _pil_to_texture(
        self, image: Image.Image, name: str, is_grayscale: bool = False
    ) -> Texture:
        """Convert PIL Image to Panda3D Texture."""
        width, height = image.size

        texture = Texture(name)

        if is_grayscale:
            # Grayscale for ID map
            pnm = PNMImage(width, height, 1)
            for y in range(height):
                for x in range(width):
                    val = image.getpixel((x, y))
                    pnm.setGray(x, y, val / 255.0)
            texture.load(pnm)
            texture.setFormat(Texture.F_red)  # Use single channel
        else:
            # RGBA for Color map
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            pnm = PNMImage(width, height, 4)
            for y in range(height):
                for x in range(width):
                    r, g, b, a = image.getpixel((x, y))
                    pnm.setXelA(x, y, r / 255.0, g / 255.0, b / 255.0, a / 255.0)
            texture.load(pnm)

        texture.setWrapU(Texture.WMRepeat)
        texture.setWrapV(Texture.WMClamp)

        return texture

    def set_vectors_visible(self, visible: bool):
        """Toggle visibility of velocity vectors."""
        if self.globe:
            val = 1.0 if visible else 0.0
            self.globe.setShaderInput("show_vectors", val)

    def set_borders_visible(self, visible: bool):
        """Toggle visibility of plate borders."""
        if self.globe:
            val = 1.0 if visible else 0.0
            self.globe.setShaderInput("show_borders", val)

    def set_elevation_visible(self, visible: bool):
        """Toggle between normal view and elevation map view."""
        self._elevation_visible = visible
        if self.globe and self._has_elevation:
            if visible:
                # Switch to elevation texture
                self.globe.setTexture(self._elevation_tex, 1)
                self.globe.setShaderInput("show_borders", 0.0)
                self.globe.setShaderInput("show_vectors", 0.0)
            else:
                # Switch back to plate texture
                self.globe.setTexture(self._color_tex, 1)

    def set_elevation_texture(self, pil_image):
        """Set the elevation map texture from a PIL Image."""
        from PIL import Image

        width, height = pil_image.size

        pnm = PNMImage(width, height, 3)
        for y in range(height):
            for x in range(width):
                r, g, b = pil_image.getpixel((x, y))[:3]
                pnm.setXel(x, y, r / 255.0, g / 255.0, b / 255.0)

        self._elevation_tex.load(pnm)
        self._has_elevation = True
        print(f"Loaded elevation texture ({width}x{height})")

    def render_velocity_vectors(self, plates: List["PlateData"], visible: bool = True):
        """
        Render velocity vectors onto the vector texture.

        Args:
            plates: List of plates with velocity vectors
            visible: Whether to show the vectors
        """
        self.set_vectors_visible(visible)
        if not visible:
            return

        # Generate texture on CPU
        img = Image.new("RGBA", (1024, 512), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Arrow shape (pointing UP, centered at 0,0)
        # Scale: 20px
        arrow_pts = [(0, -20), (10, 10), (0, 0), (-10, 10)]

        width = 1024
        height = 512

        for plate in plates:
            if plate.centroid is None or plate.velocity_vector is None:
                continue

            # 1. Get Centroid Lat/Lon
            lat, lon = self._vec3_to_latlon(plate.centroid)

            # 2. Map to UV (Pix)
            x = (lon + 180.0) / 360.0 * width
            y = (90.0 - lat) / 180.0 * height

            # 3. Calculate Azimuth
            # Convert 3D velocity to local tangent plane
            azimuth = self._calculate_azimuth(plate.centroid, plate.velocity_vector)

            # 4. Draw Arrow
            # Rotate points
            # Azimuth is clockwise from North.
            # In screen coords, Y is down.
            # North is -Y.
            # 0 deg = Up (-Y).
            # 90 deg = Right (+X).
            # cos(theta) * x - sin(theta) * y ...

            theta = math.radians(azimuth)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            # Transform points
            poly_pts = []
            for px, py in arrow_pts:
                # Rotate
                rx = px * cos_t - py * sin_t
                ry = px * sin_t + py * cos_t

                # Translate
                poly_pts.append((x + rx, y + ry))

            draw.polygon(poly_pts, fill=(255, 255, 0, 255))  # Yellow

            # Handle wrapping (if x < margin or x > width - margin)
            # Draw copy offset by width
            if x < 40:
                poly_pts_r = [(px + width, py) for px, py in poly_pts]
                draw.polygon(poly_pts_r, fill=(255, 255, 0, 255))
            elif x > width - 40:
                poly_pts_l = [(px - width, py) for px, py in poly_pts]
                draw.polygon(poly_pts_l, fill=(255, 255, 0, 255))

        # Upload to Texture
        texture = self._pil_to_texture(img, "vector_map")
        # We need to copy ram image to self._vector_tex
        self._vector_tex.setRamImage(texture.getRamImage())

    def _vec3_to_latlon(self, v: np.ndarray) -> Tuple[float, float]:
        v_norm = v / np.linalg.norm(v)
        lat = np.degrees(np.arcsin(v_norm[2]))
        lon = np.degrees(np.arctan2(v_norm[1], v_norm[0]))
        return (lat, lon)

    def _calculate_azimuth(self, pos: np.ndarray, vel: np.ndarray) -> float:
        """
        Calculate azimuth of velocity vector at position.
        Returns degrees clockwise from North.
        """
        # Normal is position (sphere)
        p = pos / np.linalg.norm(pos)

        # North pole
        k = np.array([0, 0, 1])

        # East direction: k x p
        # If p is North Pole, East is undefined.
        if abs(p[2]) > 0.99:
            east = np.array([1, 0, 0])
        else:
            east = np.cross(k, p)
            east = east / np.linalg.norm(east)

        # North direction: p x east
        north = np.cross(p, east)

        # Project velocity onto East/North
        v_e = np.dot(vel, east)
        v_n = np.dot(vel, north)

        # Atan2(y, x) -> y=East, x=North
        # Result is angle from North towards East (CW)
        angle = math.degrees(math.atan2(v_e, v_n))
        return angle
