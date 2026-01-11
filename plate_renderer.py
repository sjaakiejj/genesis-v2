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
    ) -> Optional[Tuple[Image.Image, Image.Image]]:
        """
        Generate equirectangular plate texture and ID map.

        Args:
            plates: List of PlateData with vertices and colors
            progress_callback: Optional callback for progress updates

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
        self._draw_plate_polygons(color_image, id_image, plates)

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
        self, color_img: Image.Image, id_img: Image.Image, plates: List["PlateData"]
    ):
        """Draw plate polygons onto the textures."""
        draw_color = ImageDraw.Draw(color_img)
        draw_id = ImageDraw.Draw(id_img)

        # Generate unique color map for IDs if needed, or simple grayscale
        # We assume IDs fit in 0-255 for now

        plate_id_to_index = {plate.plate_id: i for i, plate in enumerate(plates)}

        for plate in plates:
            if not plate.boundary_polygon:
                continue

            # Color for this plate
            r, g, b = plate.color
            fill_color = (int(r * 255), int(g * 255), int(b * 255))

            # ID index
            idx = plate_id_to_index.get(plate.plate_id, 255)

            # Helper to map lat/lon to pixel coords
            def to_pixels(lon, lat):
                # lon: -180 to 180 -> 0 to width
                x = (lon + 180.0) / 360.0 * self.width
                # lat: 90 to -90 -> 0 to height (flipped Y)
                y = (90.0 - lat) / 180.0 * self.height
                return x, y

            # Draw each polygon in the MultiPolygon
            for poly in plate.boundary_polygon.geoms:
                # Get exterior coordinates
                pixels = [to_pixels(*coord) for coord in poly.exterior.coords]

                # Draw filled polygon
                draw_color.polygon(pixels, fill=fill_color)
                draw_id.polygon(pixels, fill=idx)

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

    def start_generation(self, plates: List["PlateData"]):
        """
        Start texture generation in a background thread.

        Args:
            plates: List of PlateData to generate texture from
        """
        if self._is_generating:
            self.cancel()

        self._is_generating = True
        self._thread = threading.Thread(
            target=self._generation_worker, args=(plates,), daemon=True
        )
        self._thread.start()

    def _generation_worker(self, plates: List["PlateData"]):
        """Worker thread for texture generation."""
        try:
            result = self.generator.generate_texture(
                plates, progress_callback=self._on_progress
            )
            # result is tuple (color_image, id_image)
            self._result_queue.put(("success", result))
        except Exception as e:
            self._result_queue.put(("error", str(e)))
        finally:
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

        # Initialize with zeros (black)
        img = PNMImage(256, 1, 1)
        img.fill(0)
        self._selection_tex.load(img)

        # Bind selection texture to shader input
        self.globe.setShaderInput("selection_tex", self._selection_tex)

    def clear(self):
        """Remove existing globe geometry."""
        if self.globe is not None:
            self.globe.removeNode()
            self.globe = None

    def start_plate_generation(
        self, plates: List["PlateData"], selected_ids: Optional[Set[int]] = None
    ):
        """
        Start async plate texture generation.
        """
        self._plates = plates
        self._selected_ids = selected_ids or set()
        # Update selection texture immediately to match new plate list/indices
        self._update_selection_texture()
        self.threaded_generator.start_generation(plates)
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
