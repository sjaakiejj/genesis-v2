"""
Tectonic Map Generator - Main Application
A Panda3D-based globe renderer with camera orbit controls.
Uses separate display regions for 3D viewport and UI.
Supports plate selection and merging.
"""

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    WindowProperties,
    Vec3,
    Point3,
    LColor,
    AmbientLight,
    DirectionalLight,
    NodePath,
)
import math
import numpy as np

# Local modules
from tectonic_engine import PlateManager
from plate_renderer import PlateRenderer
from ui_manager import UIManager


# Sidebar takes up this fraction of the window width
SIDEBAR_RATIO = 0.22


class TectonicMapGenerator(ShowBase):
    """Main application class for the Tectonic Map Generator."""

    def __init__(self):
        super().__init__()

        # Window configuration
        self.set_window_title("Tectonic Map Generator")
        self._set_window_size(1280, 800)

        # Disable default mouse camera control
        self.disableMouse()

        # Camera orbit parameters
        self.camera_distance = 6.0
        self.camera_heading = 0.0
        self.camera_pitch = 20.0
        self.orbit_speed = 0.3

        # Globe parameters
        self.globe_radius = 2.0  # Scale of the rendered globe

        # Mouse tracking
        self.mouse_dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        # Generation state
        self._is_generating = False
        self._selection_refresh_pending = False

        # Setup display regions first
        self._setup_display_regions()

        # Setup components
        self._setup_lighting()
        self._setup_globe()
        self._setup_ui()
        self._setup_camera()
        self._setup_input()

        # Add update task
        self.taskMgr.add(self._update_task, "update_task")

    def set_window_title(self, title: str):
        """Set the window title."""
        props = WindowProperties()
        props.setTitle(title)
        self.win.requestProperties(props)

    def _set_window_size(self, width: int, height: int):
        """Set the window size."""
        props = WindowProperties()
        props.setSize(width, height)
        self.win.requestProperties(props)

    def _setup_display_regions(self):
        """Setup separate display regions for 3D viewport and UI."""
        viewport_ratio = 1.0 - SIDEBAR_RATIO

        dr = self.cam.node().getDisplayRegion(0)
        dr.setDimensions(0, viewport_ratio, 0, 1)

        self._viewport_ratio = viewport_ratio

        window_aspect = self.getAspectRatio()
        new_aspect = window_aspect * viewport_ratio
        self.cam.node().getLens().setAspectRatio(new_aspect)

    def _setup_lighting(self):
        """Setup scene lighting with per-pixel shading."""
        self.render.setShaderAuto()

        ambient_light = AmbientLight("ambient")
        ambient_light.setColor(LColor(0.2, 0.2, 0.25, 1.0))
        ambient_np = self.render.attachNewNode(ambient_light)
        self.render.setLight(ambient_np)

        key_light = DirectionalLight("key_light")
        key_light.setColor(LColor(1.0, 0.95, 0.9, 1.0))
        key_np = self.render.attachNewNode(key_light)
        key_np.setHpr(45, -45, 0)
        self.render.setLight(key_np)

        fill_light = DirectionalLight("fill_light")
        fill_light.setColor(LColor(0.3, 0.35, 0.4, 1.0))
        fill_np = self.render.attachNewNode(fill_light)
        fill_np.setHpr(-135, 30, 0)
        self.render.setLight(fill_np)

    def _setup_globe(self):
        """Setup the globe with plate renderer (initially blank)."""
        self.plate_manager = PlateManager(num_plates=12)
        self.globe_center = Point3(0, 0, 0)
        self.plate_renderer = PlateRenderer(
            parent_node=self.render, scale=self.globe_radius
        )
        self.plate_renderer.setup_globe()

    def _setup_ui(self):
        """Setup the UI sidebar."""
        self.ui_manager = UIManager(self, sidebar_ratio=SIDEBAR_RATIO)
        self.ui_manager.set_generate_callback(self.generate_map)
        self.ui_manager.set_merge_callback(self.merge_selected_plates)

    def _setup_camera(self):
        """Setup camera position and orientation."""
        self._update_camera_position()

    def _update_camera_position(self):
        """Update camera position based on orbit parameters."""
        heading_rad = math.radians(self.camera_heading)
        pitch_rad = math.radians(self.camera_pitch)
        pitch_rad = max(-math.pi / 2 + 0.1, min(math.pi / 2 - 0.1, pitch_rad))

        x = self.camera_distance * math.cos(pitch_rad) * math.sin(heading_rad)
        y = -self.camera_distance * math.cos(pitch_rad) * math.cos(heading_rad)
        z = self.camera_distance * math.sin(pitch_rad)

        self.camera.setPos(x, y, z)
        self.camera.lookAt(self.globe_center)

    def _setup_input(self):
        """Setup keyboard and mouse input bindings."""
        self.accept("space", self.generate_map)
        self.accept("escape", self._quit_application)
        self.accept("mouse1", self._on_mouse_press)
        self.accept("mouse1-up", self._on_mouse_release)
        self.accept("mouse3", self._on_right_click)  # Right-click for selection
        self.accept("wheel_up", self._zoom_in)
        self.accept("wheel_down", self._zoom_out)
        self.accept("window-event", self._on_window_event)

    def _on_window_event(self, window):
        """Handle window resize events to update aspect ratio."""
        if window == self.win:
            self._setup_display_regions()

    def _on_mouse_press(self):
        """Handle left mouse button press for camera orbit."""
        if self.mouseWatcherNode.hasMouse():
            mouse_x = self.mouseWatcherNode.getMouseX()
            if mouse_x < self._viewport_ratio * 2 - 1:
                self.mouse_dragging = True
                self.last_mouse_x = mouse_x
                self.last_mouse_y = self.mouseWatcherNode.getMouseY()

    def _on_mouse_release(self):
        """Handle mouse button release."""
        self.mouse_dragging = False

    def _on_right_click(self):
        """Handle right-click for plate selection."""
        if not self.mouseWatcherNode.hasMouse():
            return

        # Check if plates exist
        if not self.plate_manager.plates:
            return

        mouse_x = self.mouseWatcherNode.getMouseX()
        mouse_y = self.mouseWatcherNode.getMouseY()

        # Check if click is in the 3D viewport
        if mouse_x >= self._viewport_ratio * 2 - 1:
            return  # Click was in sidebar

        # Perform ray-sphere intersection
        hit_point = self._raycast_to_globe(mouse_x, mouse_y)

        if hit_point is not None:
            # Find which plate was clicked
            plate_id = self.plate_manager.get_plate_at_point(hit_point)

            if plate_id is not None:
                # Toggle selection
                is_now_selected = self.plate_manager.toggle_selection(plate_id)
                action = "Selected" if is_now_selected else "Deselected"
                print(f"{action} plate {plate_id}")

                # Update UI
                self._update_selection_ui()

                # Refresh texture to show highlighting
                self._refresh_selection_highlight()

    def _raycast_to_globe(self, mouse_x: float, mouse_y: float) -> np.ndarray:
        """
        Cast a ray from camera through mouse position and intersect with globe sphere.

        Args:
            mouse_x: Mouse x in normalized device coords (-1 to 1)
            mouse_y: Mouse y in normalized device coords (-1 to 1)

        Returns:
            3D hit point on sphere surface, or None if no intersection
        """
        # Get camera position and lens
        cam_pos = self.camera.getPos()
        lens = self.cam.node().getLens()

        # Get ray direction from camera through mouse point
        # Map window coordinates [-1, 1] to viewport coordinates [-1, 1]
        # logic: (mouse_x + 1) converts to [0, 2]
        # / viewport_ratio scales to [0, 2/ratio] (relative to viewport/window size)
        # - 1 shifts back to [-1, ...] range
        adjusted_x = (mouse_x + 1) / self._viewport_ratio - 1

        # Use lens to compute ray
        near_point = Point3()
        far_point = Point3()

        if not lens.extrude(Point3(adjusted_x, mouse_y, 0), near_point, far_point):
            return None

        # Transform to world space
        near_world = self.render.getRelativePoint(self.cam, near_point)
        far_world = self.render.getRelativePoint(self.cam, far_point)

        # Ray direction
        ray_origin = np.array([near_world.x, near_world.y, near_world.z])
        ray_dir = np.array(
            [
                far_world.x - near_world.x,
                far_world.y - near_world.y,
                far_world.z - near_world.z,
            ]
        )
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        # Sphere at origin with radius = globe_radius
        sphere_center = np.array([0.0, 0.0, 0.0])
        sphere_radius = self.globe_radius

        # Ray-sphere intersection
        # Solve: |ray_origin + t * ray_dir - sphere_center|^2 = sphere_radius^2
        oc = ray_origin - sphere_center
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - sphere_radius * sphere_radius

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None

        # Take the closer intersection (smaller t)
        t = (-b - math.sqrt(discriminant)) / (2.0 * a)

        if t < 0:
            # Try the other intersection
            t = (-b + math.sqrt(discriminant)) / (2.0 * a)
            if t < 0:
                return None

        # Compute hit point
        hit_point = ray_origin + t * ray_dir

        # Normalize to unit sphere for plate lookup (PlateManager uses radius=1)
        hit_point_normalized = hit_point / self.globe_radius

        return hit_point_normalized

    def _update_selection_ui(self):
        """Update UI to reflect current selection state."""
        count = self.plate_manager.get_selection_count()
        can_merge = self.plate_manager.are_selected_neighbors()
        self.ui_manager.update_selection_state(count, can_merge)

    def _refresh_selection_highlight(self):
        """Regenerate texture with current selection highlighting."""
        if self._is_generating:
            # Mark that we need to refresh after current generation
            self._selection_refresh_pending = True
            return

        selected_ids = self.plate_manager.get_selected_ids()
        self.plate_renderer.refresh_selection(selected_ids)
        self._is_generating = True
        self.ui_manager.set_status("Updating selection...")

    def _zoom_in(self):
        """Zoom camera in."""
        self.camera_distance = max(3.0, self.camera_distance - 0.5)
        self._update_camera_position()

    def _zoom_out(self):
        """Zoom camera out."""
        self.camera_distance = min(30.0, self.camera_distance + 0.5)
        self._update_camera_position()

    def _update_task(self, task: Task) -> int:
        """Main update loop task."""
        # Handle mouse orbit
        if self.mouse_dragging and self.mouseWatcherNode.hasMouse():
            mouse_x = self.mouseWatcherNode.getMouseX()
            mouse_y = self.mouseWatcherNode.getMouseY()

            delta_x = mouse_x - self.last_mouse_x
            delta_y = mouse_y - self.last_mouse_y

            self.camera_heading += delta_x * 100 * self.orbit_speed
            self.camera_pitch -= delta_y * 100 * self.orbit_speed
            self.camera_pitch = max(-89, min(89, self.camera_pitch))

            self._update_camera_position()

            self.last_mouse_x = mouse_x
            self.last_mouse_y = mouse_y

        # Update generation progress
        if self._is_generating:
            progress = self.plate_renderer.update()

            if progress:
                self.ui_manager.update_progress(progress.percentage, progress.message)

            # Check if generation completed
            if not self.plate_renderer.is_generating():
                self._is_generating = False
                self.ui_manager.set_generating(False)
                self.ui_manager.set_status("Right-click plates to select")

                # Update texture preview
                texture = self.plate_renderer.get_current_texture()
                if texture:
                    self.ui_manager.update_texture_preview(texture)

                # Handle pending selection refresh
                if self._selection_refresh_pending:
                    self._selection_refresh_pending = False
                    self._refresh_selection_highlight()

        return Task.cont

    def _quit_application(self):
        """Quit the application."""
        print("Exiting Tectonic Map Generator...")
        self.userExit()

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def generate_map(self):
        """Generate a new tectonic map."""
        if self._is_generating:
            return

        print("Starting plate generation...")
        self._is_generating = True
        self.ui_manager.set_generating(True)
        self.ui_manager.set_status("Generating tectonic plates...")

        # Clear selection when regenerating
        self.plate_manager.clear_selection()
        self._update_selection_ui()

        # Generate plate data
        plates = self.plate_manager.generate()

        # Start async texture generation
        self.plate_renderer.start_plate_generation(plates)

    def merge_selected_plates(self):
        """Merge the currently selected plates."""
        if self._is_generating:
            return

        if not self.plate_manager.are_selected_neighbors():
            print("Cannot merge: selected plates are not neighbors")
            return

        print("Merging selected plates...")

        # Perform merge
        if self.plate_manager.merge_selected_plates():
            # Update UI
            self._update_selection_ui()

            # Regenerate texture with updated plates
            self._is_generating = True
            self.ui_manager.set_status("Regenerating after merge...")
            self.plate_renderer.start_plate_generation(
                self.plate_manager.plates, self.plate_manager.get_selected_ids()
            )


def main():
    """Entry point for the Tectonic Map Generator application."""
    app = TectonicMapGenerator()
    app.run()


if __name__ == "__main__":
    main()
