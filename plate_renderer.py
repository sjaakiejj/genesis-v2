"""
Plate Renderer - Panda3D Visualization Module
Renders plates as a texture applied to a procedural sphere.
Supports threaded generation with progress callbacks and selection highlighting.
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
)
import numpy as np
from PIL import Image
from typing import List, Callable, Optional, Set, TYPE_CHECKING
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
    Supports threaded execution with progress callbacks and selection highlighting.
    """

    # Selection highlight settings
    HIGHLIGHT_BRIGHTNESS = 1.4  # Brighten selected plates
    HIGHLIGHT_BORDER_WIDTH = 3  # Pixels for border effect

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
        selected_ids: Optional[Set[int]] = None,
        progress_callback: Optional[Callable[[GenerationProgress], None]] = None,
    ) -> Optional[Image.Image]:
        """
        Generate an equirectangular plate texture with optional selection highlighting.

        Args:
            plates: List of PlateData with vertices and colors
            selected_ids: Set of plate IDs to highlight
            progress_callback: Optional callback for progress updates

        Returns:
            PIL Image with plate colors, or None if cancelled
        """
        self._cancel_flag = False
        selected_ids = selected_ids or set()

        # Create image with ocean blue background
        image = Image.new("RGB", (self.width, self.height), (30, 60, 90))
        pixels = image.load()

        # Build plate ID map for each pixel (for border detection)
        plate_id_map = np.zeros((self.height, self.width), dtype=np.int32)

        # Precompute plate centroids for faster lookup
        centroids = []
        for plate in plates:
            if plate.centroid is not None:
                centroid = plate.centroid / np.linalg.norm(plate.centroid)
            else:
                centroid = np.mean(plate.vertices, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
            centroids.append((plate, centroid))

        total_rows = self.height

        # Pass 1: Assign plate IDs and colors to each pixel
        for y in range(self.height):
            if self._cancel_flag:
                return None

            if progress_callback and y % 20 == 0:
                progress = GenerationProgress(
                    current=y,
                    total=total_rows,
                    percentage=(y / total_rows) * 100,
                    message=f"Generating texture row {y}/{total_rows}",
                )
                progress_callback(progress)

            lat = 90.0 - (y / self.height) * 180.0
            lat_rad = math.radians(lat)
            cos_lat = math.cos(lat_rad)
            sin_lat = math.sin(lat_rad)

            for x in range(self.width):
                lon = (x / self.width) * 360.0 - 180.0
                lon_rad = math.radians(lon)

                px = cos_lat * math.cos(lon_rad)
                py = cos_lat * math.sin(lon_rad)
                pz = sin_lat
                point = np.array([px, py, pz])

                closest_plate = self._find_closest_plate(point, centroids)

                if closest_plate is not None:
                    plate_id_map[y, x] = closest_plate.plate_id
                    r, g, b = closest_plate.color

                    # Highlight selected plates
                    if closest_plate.plate_id in selected_ids:
                        r = min(1.0, r * self.HIGHLIGHT_BRIGHTNESS)
                        g = min(1.0, g * self.HIGHLIGHT_BRIGHTNESS)
                        b = min(1.0, b * self.HIGHLIGHT_BRIGHTNESS)

                    pixels[x, y] = (int(r * 255), int(g * 255), int(b * 255))

        # Pass 2: Add borders around selected plates
        if selected_ids:
            self._add_selection_borders(image, plate_id_map, selected_ids)

        if progress_callback:
            progress = GenerationProgress(
                current=total_rows,
                total=total_rows,
                percentage=100.0,
                message="Texture generation complete",
            )
            progress_callback(progress)

        return image

    def _add_selection_borders(
        self, image: Image.Image, plate_id_map: np.ndarray, selected_ids: Set[int]
    ):
        """Add white borders around selected plates."""
        pixels = image.load()
        height, width = plate_id_map.shape
        border_color = (255, 255, 255)  # White border

        for y in range(height):
            for x in range(width):
                plate_id = plate_id_map[y, x]
                if plate_id not in selected_ids:
                    continue

                # Check if this pixel is on a plate boundary
                is_border = False
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, (x + dx) % width  # Wrap x for equirectangular
                        if 0 <= ny < height:
                            neighbor_id = plate_id_map[ny, nx]
                            if neighbor_id != plate_id:
                                is_border = True
                                break
                    if is_border:
                        break

                if is_border:
                    pixels[x, y] = border_color

    def cancel(self):
        """Cancel ongoing generation."""
        self._cancel_flag = True

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

    def start_generation(
        self, plates: List["PlateData"], selected_ids: Optional[Set[int]] = None
    ):
        """
        Start texture generation in a background thread.

        Args:
            plates: List of PlateData to generate texture from
            selected_ids: Set of plate IDs to highlight
        """
        if self._is_generating:
            self.cancel()

        self._is_generating = True
        self._thread = threading.Thread(
            target=self._generation_worker, args=(plates, selected_ids), daemon=True
        )
        self._thread.start()

    def _generation_worker(
        self, plates: List["PlateData"], selected_ids: Optional[Set[int]]
    ):
        """Worker thread for texture generation."""
        try:
            image = self.generator.generate_texture(
                plates, selected_ids=selected_ids, progress_callback=self._on_progress
            )
            self._result_queue.put(("success", image))
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
    Renders tectonic plates using a textured sphere.
    """

    def __init__(self, parent_node: NodePath, scale: float = 2.0):
        """
        Initialize the plate renderer.

        Args:
            parent_node: Panda3D NodePath to parent geometry to
            scale: Scale factor for the globe
        """
        self.parent_node = parent_node
        self.scale = scale
        self.globe: NodePath = None
        self.threaded_generator = ThreadedTextureGenerator()
        self._current_texture: Optional[Image.Image] = None
        self._plates: List["PlateData"] = []
        self._selected_ids: Set[int] = set()

    def setup_globe(self):
        """Create the globe sphere without any texture (blank state)."""
        if self.globe is not None:
            self.globe.removeNode()

        self.globe = self._create_uv_sphere(segments=64, rings=48)
        self.globe.setScale(self.scale)
        self.globe.reparentTo(self.parent_node)

        # Set a default ocean blue color
        self.globe.setColor(LColor(0.12, 0.24, 0.35, 1.0))

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

        Args:
            plates: List of PlateData from PlateManager
            selected_ids: Optional set of plate IDs to highlight
        """
        self._plates = plates
        self._selected_ids = selected_ids or set()
        self.threaded_generator.start_generation(plates, selected_ids)
        print(f"Started texture generation for {len(plates)} plates...")

    def refresh_selection(self, selected_ids: Set[int]):
        """
        Regenerate texture with updated selection highlighting.

        Args:
            selected_ids: Set of plate IDs to highlight
        """
        if self._plates:
            self._selected_ids = selected_ids
            self.threaded_generator.start_generation(self._plates, selected_ids)

    def update(self) -> Optional[GenerationProgress]:
        """
        Update method to be called each frame.
        Checks for completed texture and applies it.

        Returns:
            Current progress if generating, None otherwise
        """
        progress = self.threaded_generator.get_progress()

        result = self.threaded_generator.get_result()
        if result is not None:
            status, data = result
            if status == "success" and data is not None:
                self._current_texture = data
                self._apply_texture(data)
                print("Texture applied to globe")
            elif status == "error":
                print(f"Texture generation error: {data}")

        return progress

    def is_generating(self) -> bool:
        """Check if texture generation is in progress."""
        return self.threaded_generator.is_generating()

    def get_current_texture(self) -> Optional[Image.Image]:
        """Get the current plate texture image."""
        return self._current_texture

    def get_plates(self) -> List["PlateData"]:
        """Get the current plates list."""
        return self._plates

    def _apply_texture(self, image: Image.Image):
        """Apply a PIL image as texture to the globe."""
        if self.globe is None:
            self.setup_globe()

        texture = self._pil_to_texture(image)
        self.globe.setTexture(texture)
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

    def _pil_to_texture(self, image: Image.Image) -> Texture:
        """Convert PIL Image to Panda3D Texture."""
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        width, height = image.size

        pnm = PNMImage(width, height, 4)

        for y in range(height):
            for x in range(width):
                r, g, b, a = image.getpixel((x, y))
                pnm.setXelA(x, y, r / 255.0, g / 255.0, b / 255.0, a / 255.0)

        texture = Texture("plate_texture")
        texture.load(pnm)
        texture.setWrapU(Texture.WMRepeat)
        texture.setWrapV(Texture.WMClamp)

        return texture
