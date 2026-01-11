"""
Tectonic Engine - Core Logic Module
Uses scipy for geometry and pygplates for data structures.
"""

import numpy as np
from scipy.spatial import SphericalVoronoi
import shapely
import shapely.affinity
import pygplates
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict
import random
import colorsys
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import clip_by_rect


@dataclass
class PlateData:
    """Lightweight data class for plate visualization."""

    plate_id: int
    vertices: np.ndarray  # Nx3 array of 3D vertices
    color: Tuple[float, float, float]  # RGB (0-1)
    centroid: np.ndarray = None  # 3D centroid point
    seed_point: np.ndarray = None  # 3D seed point for Voronoi generation
    boundary_polygon: Optional[MultiPolygon] = None  # Shapely MultiPolygon (lat/lon)


class PlateManager:
    """
    Manages tectonic plates using pygplates data structures.

    Converts scipy SphericalVoronoi regions to pygplates Features
    with PolygonOnSphere geometry.
    """

    def __init__(
        self, num_plates: int = 12, radius: float = 1.0, seed: Optional[int] = None
    ):
        """
        Initialize the plate manager.

        Args:
            num_plates: Number of tectonic plates to generate
            radius: Sphere radius (default 1.0)
            seed: Random seed for reproducibility
        """
        self.num_plates = num_plates
        self.radius = radius
        self.seed = seed

        # pygplates feature collection
        self.feature_collection: pygplates.FeatureCollection = None

        # Plate data for visualization
        self.plates: List[PlateData] = []

        # Selection state
        self._selected_ids: Set[int] = set()

        # Neighbor map (plate_id -> set of neighbor plate_ids)
        self._neighbors: Dict[int, Set[int]] = {}

        # Underlying Voronoi tessellation
        self._voronoi: SphericalVoronoi = None

        # Map from plate_id to PlateData
        self._plate_map: Dict[int, PlateData] = {}

    def generate(self) -> List[PlateData]:
        """
        Generate tectonic plates.

        1. Generate seed points on sphere
        2. Create SphericalVoronoi tessellation
        3. Convert regions to pygplates Features
        4. Return plate data for visualization

        Returns:
            List of PlateData for rendering
        """
        # Clear selection when regenerating
        self.clear_selection()

        # Set random seed
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
        else:
            np.random.seed()
            random.seed()

        # Generate seed points
        seed_points = self._generate_seed_points(self.num_plates)

        # Create Voronoi tessellation
        self._voronoi = SphericalVoronoi(
            points=seed_points, radius=self.radius, center=np.array([0.0, 0.0, 0.0])
        )
        self._voronoi.sort_vertices_of_regions()

        # Create pygplates feature collection
        self._create_pygplates_features(seed_points)

        # Create plate data for visualization
        self._create_plate_data()

        # Build neighbor map from Voronoi adjacency
        self._build_neighbor_map()

        print(f"Generated {len(self.plates)} tectonic plates with pygplates Features")
        return self.plates

    def _generate_seed_points(self, n: int) -> np.ndarray:
        """Generate n uniformly distributed points on sphere."""
        points = np.random.randn(n, 3)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms * self.radius

    def _create_pygplates_features(self, seed_points: np.ndarray):
        """
        Create pygplates Features from Voronoi regions.

        Each region becomes a Feature with:
        - PolygonOnSphere geometry
        - Random PlateID
        """
        features = []

        for i, region_indices in enumerate(self._voronoi.regions):
            if len(region_indices) < 3:
                continue

            # Get vertices for this region
            vertices = self._voronoi.vertices[region_indices]

            # Convert 3D Cartesian to (lat, lon) for pygplates
            lat_lon_points = []
            for v in vertices:
                v_norm = v / np.linalg.norm(v)
                lat = np.degrees(np.arcsin(v_norm[2]))
                lon = np.degrees(np.arctan2(v_norm[1], v_norm[0]))
                lat_lon_points.append(pygplates.PointOnSphere(lat, lon))

            # Create PolygonOnSphere from points
            try:
                polygon = pygplates.PolygonOnSphere(lat_lon_points)
            except pygplates.InvalidLatLonError:
                print(f"Warning: Invalid polygon for region {i}, skipping")
                continue

            # Create feature with polygon geometry
            feature = pygplates.Feature()
            feature.set_geometry(polygon)

            # Assign random plate ID (integer between 100 and 999)
            plate_id = random.randint(100, 999)
            feature.set_reconstruction_plate_id(plate_id)

            features.append(feature)

        self.feature_collection = pygplates.FeatureCollection(features)
        print(f"Created pygplates FeatureCollection with {len(features)} features")

    def _create_plate_data(self):
        """Create PlateData objects from Voronoi regions for visualization."""
        self.plates = []
        self._plate_map = {}

        for i, region_indices in enumerate(self._voronoi.regions):
            if len(region_indices) < 3:
                continue

            # Get vertices
            vertices = self._voronoi.vertices[region_indices]

            # Normalize to sphere surface
            norms = np.linalg.norm(vertices, axis=1, keepdims=True)
            vertices = vertices / norms * self.radius

            # Compute centroid
            centroid = np.mean(vertices, axis=0)
            centroid = centroid / np.linalg.norm(centroid) * self.radius

            # Generate color
            color = self._generate_plate_color()

            # Get original seed point
            seed_point = self._voronoi.points[i]

            plate = PlateData(
                plate_id=i,
                vertices=vertices,
                color=color,
                centroid=centroid,
                seed_point=seed_point,
            )
            # Create explicit polygon boundary
            plate.boundary_polygon = self._create_polygon_from_vertices(vertices)

            self.plates.append(plate)
            self._plate_map[i] = plate

    def _build_neighbor_map(self):
        """Build neighbor relationships from Voronoi ridge data."""
        self._neighbors = {plate.plate_id: set() for plate in self.plates}

        # Get valid plate IDs
        valid_ids = set(self._plate_map.keys())

        # SphericalVoronoi doesn't directly expose ridges, so we compute neighbors
        # by finding regions that share vertices
        regions = self._voronoi.regions

        # Build vertex -> region mapping
        vertex_to_regions: Dict[int, Set[int]] = {}
        for region_idx, region_vertices in enumerate(regions):
            if region_idx not in valid_ids:
                continue
            for vertex_idx in region_vertices:
                if vertex_idx not in vertex_to_regions:
                    vertex_to_regions[vertex_idx] = set()
                vertex_to_regions[vertex_idx].add(region_idx)

        # Regions sharing a vertex are neighbors
        for vertex_idx, region_set in vertex_to_regions.items():
            region_list = list(region_set)
            for i, r1 in enumerate(region_list):
                for r2 in region_list[i + 1 :]:
                    if r1 in self._neighbors and r2 in self._neighbors:
                        self._neighbors[r1].add(r2)
                        self._neighbors[r2].add(r1)

    def _generate_plate_color(self) -> Tuple[float, float, float]:
        """Generate a vibrant random color using HSV."""
        hue = random.random()
        saturation = 0.5 + random.random() * 0.3
        value = 0.6 + random.random() * 0.3
        return colorsys.hsv_to_rgb(hue, saturation, value)

    def _create_polygon_from_vertices(self, vertices: np.ndarray) -> MultiPolygon:
        """
        Convert 3D vertices to a Shapely MultiPolygon on equirectangular projection.
        Handles dateline wrapping.
        """
        # Convert to lat/lon with standard spherical interpolation (SLERP)
        # to ensure edges follow great circles
        coords = []
        num_verts = len(vertices)

        for i in range(num_verts):
            v_start = vertices[i]
            v_end = vertices[(i + 1) % num_verts]

            # Calculate angle between vectors
            v1_norm = v_start / np.linalg.norm(v_start)
            v2_norm = v_end / np.linalg.norm(v_end)
            dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            angle = np.arccos(dot)

            # Number of steps based on angle (e.g., 1 step per 2 degrees)
            steps = max(1, int(np.degrees(angle) / 2.0))

            for t in np.linspace(0, 1, steps, endpoint=False):
                v = self._slerp(v_start, v_end, t)
                v_norm = v / np.linalg.norm(v)

                lat = np.degrees(np.arcsin(v_norm[2]))
                lon = np.degrees(np.arctan2(v_norm[1], v_norm[0]))
                coords.append((lon, lat))

        # Close the loop logic handled by wrapping, but we need to ensure the polygon is closed
        # Since we used endpoint=False and loop over all edges, the first point of the next edge
        # effectively closes the previous edge.
        pass  # loop structure handles it

        # However, the previous code structure was:
        # for v in vertices: convert...
        # append(coords[0]) to close

        # New structure:
        # for each edge: interpolate points (excluding end to avoid dupe)
        # But for the last edge, we need to make sure we connect back.

        # Let's refine the loop:
        coords = []
        for i in range(num_verts):
            v_start = vertices[i]
            v_end = vertices[(i + 1) % num_verts]

            v1_norm = v_start / np.linalg.norm(v_start)
            v2_norm = v_end / np.linalg.norm(v_end)
            dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            angle = np.arccos(dot)

            steps = max(1, int(np.degrees(angle) / 2.0))

            # Interpolate (include start, exclude end)
            for j in range(steps):
                t = j / steps
                v = self._slerp(v_start, v_end, t)
                v_norm = v / np.linalg.norm(v)
                lat = np.degrees(np.arcsin(v_norm[2]))
                lon = np.degrees(np.arctan2(v_norm[1], v_norm[0]))
                coords.append((lon, lat))

        # Close loop
        if coords:
            coords.append(coords[0])

        # Simple heuristic for dateline crossing:
        # If consecutive points are far apart in longitude (> 180), we have a wrap.
        # However, for correct splitting, it's easier to:
        # 1. Shift longitudes to 0..360 if we detect crossing (making it continuous)
        # 2. Or assume it's a valid polygon and "clip" it against the global map?

        # Strategy: Project to -180..180. If edge spans > 180, unwrap one point.
        # Then we might have a polygon outside [-180, 180].
        # Finally, split/wrap it back into the box.

        unwrapped = []
        if coords:
            unwrapped.append(coords[0])
            for i in range(1, len(coords)):
                prev_lon, prev_lat = unwrapped[-1]
                curr_lon, curr_lat = coords[i]

                # Check delta
                delta_lon = curr_lon - prev_lon
                if delta_lon > 180.0:
                    curr_lon -= 360.0
                elif delta_lon < -180.0:
                    curr_lon += 360.0

                unwrapped.append((curr_lon, curr_lat))

        # Check for pole wrapping
        # If the net longitude change is ~360, we circled a pole.
        if len(unwrapped) > 1:
            start_lon = unwrapped[0][0]
            end_lon = unwrapped[-1][0]
            net_lon = end_lon - start_lon

            if abs(net_lon) > 180.0:
                # We wrapped around the globe!
                # Determine which pole: check average latitude
                avg_lat = np.mean([p[1] for p in unwrapped])

                pole_lat = 90.0 if avg_lat > 0 else -90.0

                # We need to close the loop along the pole.
                # The path goes Start -> ... -> End.
                # To close it, we go End -> (End.x, Pole) -> (Start.x, Pole) -> Start.

                unwrapped.append((end_lon, pole_lat))
                unwrapped.append((start_lon, pole_lat))

        poly = Polygon(unwrapped)

        # Buffer 0 to fix self-intersections if any
        if not poly.is_valid:
            poly = poly.buffer(0)

        # Now clip against the standard world bounds [-180, -90, 180, 90]
        # But wait, our polygon might be shifted to e.g. [170, 190].
        # We need to replicate it shifted by +/- 360 and then clip everything.

        polys = []

        # Create copies shifted by -360, 0, +360
        for shift in [-360.0, 0.0, 360.0]:
            shifted_poly = self._shift_polygon(poly, shift)
            # Clip against world box
            # bounds: minx, miny, maxx, maxy
            try:
                # clip_by_rect is efficient
                clipped = clip_by_rect(shifted_poly, -180.0, -90.0, 180.0, 90.0)
                if not clipped.is_empty:
                    if isinstance(clipped, Polygon):
                        polys.append(clipped)
                    elif isinstance(clipped, MultiPolygon):
                        for p in clipped.geoms:
                            polys.append(p)
            except Exception as e:
                # Fallback or ignore
                pass

        return MultiPolygon(polys)

    def _shift_polygon(self, poly: Polygon, x_shift: float) -> Polygon:
        """Helper to shift polygon coordinates."""
        return shapely.affinity.translate(poly, xoff=x_shift)

    def _slerp(self, v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between two vectors."""
        # Assume v0 and v1 are not normalized but have same length (radius)
        # Normalize for rotation calculation
        v0_norm = v0 / np.linalg.norm(v0)
        v1_norm = v1 / np.linalg.norm(v1)

        dot = np.clip(np.dot(v0_norm, v1_norm), -1.0, 1.0)

        # If vectors are very close, linear int is fine
        if dot > 0.9995:
            return v0 + t * (v1 - v0)

        theta = np.arccos(dot)
        sin_theta = np.sin(theta)

        if sin_theta == 0:
            return v0

        w1 = np.sin((1 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta

        return w1 * v0 + w2 * v1

    # =========================================================================
    # SELECTION API
    # =========================================================================

    def select_plate(self, plate_id: int) -> bool:
        """
        Select a plate by ID.

        Args:
            plate_id: ID of plate to select

        Returns:
            True if plate exists and was selected
        """
        if plate_id in self._plate_map:
            self._selected_ids.add(plate_id)
            return True
        return False

    def deselect_plate(self, plate_id: int):
        """Deselect a plate by ID."""
        self._selected_ids.discard(plate_id)

    def toggle_selection(self, plate_id: int) -> bool:
        """
        Toggle plate selection.

        Returns:
            True if plate is now selected, False if deselected
        """
        if plate_id in self._selected_ids:
            self._selected_ids.discard(plate_id)
            return False
        else:
            if plate_id in self._plate_map:
                self._selected_ids.add(plate_id)
                return True
        return False

    def clear_selection(self):
        """Clear all selections."""
        self._selected_ids.clear()

    def get_selected_ids(self) -> Set[int]:
        """Get set of selected plate IDs."""
        return self._selected_ids.copy()

    def get_selection_count(self) -> int:
        """Get number of selected plates."""
        return len(self._selected_ids)

    def is_selected(self, plate_id: int) -> bool:
        """Check if a plate is selected."""
        return plate_id in self._selected_ids

    # =========================================================================
    # NEIGHBOR API
    # =========================================================================

    def get_neighbors(self, plate_id: int) -> Set[int]:
        """Get neighboring plate IDs for a plate."""
        return self._neighbors.get(plate_id, set()).copy()

    def are_neighbors(self, id_a: int, id_b: int) -> bool:
        """Check if two plates are neighbors."""
        return id_b in self._neighbors.get(id_a, set())

    def are_selected_neighbors(self) -> bool:
        """
        Check if all selected plates form a connected group of neighbors.

        Returns:
            True if >=2 plates selected and they are all connected
        """
        if len(self._selected_ids) < 2:
            return False

        # Check connectivity using BFS
        selected = list(self._selected_ids)
        visited = {selected[0]}
        queue = [selected[0]]

        while queue:
            current = queue.pop(0)
            for neighbor in self._neighbors.get(current, set()):
                if neighbor in self._selected_ids and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == len(self._selected_ids)

    # =========================================================================
    # MERGE API
    # =========================================================================

    def merge_selected_plates(self) -> bool:
        """
        Merge all selected plates into one.

        Returns:
            True if merge was successful
        """
        if not self.are_selected_neighbors():
            print("Cannot merge: selected plates are not all neighbors")
            return False

        selected = list(self._selected_ids)
        if len(selected) < 2:
            return False

        # Find the largest plate to keep its color
        largest_id = max(selected, key=lambda pid: len(self._plate_map[pid].vertices))
        keep_color = self._plate_map[largest_id].color

        # Combine all vertices (simplified approach - use convex hull later if needed)
        all_vertices = []
        for pid in selected:
            all_vertices.extend(self._plate_map[pid].vertices.tolist())

        # Compute new centroid
        combined_vertices = np.array(all_vertices)
        new_centroid = np.mean(combined_vertices, axis=0)
        new_centroid = new_centroid / np.linalg.norm(new_centroid) * self.radius

        # Create merged plate with first selected ID
        new_id = selected[0]
        merged_plate = PlateData(
            plate_id=new_id,
            vertices=combined_vertices,
            color=keep_color,
            centroid=new_centroid,
        )

        # Remove old plates
        self.plates = [p for p in self.plates if p.plate_id not in selected]
        for pid in selected:
            del self._plate_map[pid]

        # Add merged plate
        self.plates.append(merged_plate)
        self._plate_map[new_id] = merged_plate

        # Update neighbors - merged plate inherits all neighbors
        merged_neighbors = set()
        for pid in selected:
            if pid in self._neighbors:
                merged_neighbors.update(self._neighbors[pid])
                del self._neighbors[pid]

        # Remove self-references
        merged_neighbors -= set(selected)
        self._neighbors[new_id] = merged_neighbors

        # Update other plates' neighbor lists
        for pid in list(self._neighbors.keys()):
            self._neighbors[pid] -= set(selected)
            if any(s in selected for s in self._neighbors.get(pid, set())):
                self._neighbors[pid].add(new_id)
            if new_id in merged_neighbors:
                self._neighbors[pid].add(new_id)

        # Clear selection
        self.clear_selection()

        print(f"Merged {len(selected)} plates into plate {new_id}")
        return True

    # =========================================================================
    # UTILITY
    # =========================================================================

    def get_plate_at_point(self, point_3d: np.ndarray) -> Optional[int]:
        """
        Find which plate contains or is closest to a 3D point.

        Args:
            point_3d: 3D point on or near the sphere

        Returns:
            plate_id of closest plate, or None if no plates
        """
        if not self.plates:
            return None

        # Normalize point to sphere surface
        point = point_3d / np.linalg.norm(point_3d) * self.radius

        # Find closest centroid
        closest_id = None
        min_dist = float("inf")

        for plate in self.plates:
            if plate.centroid is None:
                continue
            # Use dot product for angular distance
            dot = np.dot(point, plate.centroid)
            dist = -dot  # Higher dot = closer
            if dist < min_dist:
                min_dist = dist
                closest_id = plate.plate_id

        return closest_id

    @property
    def voronoi(self) -> SphericalVoronoi:
        """Access the underlying SphericalVoronoi."""
        return self._voronoi

    def get_feature_collection(self) -> pygplates.FeatureCollection:
        """Get the pygplates FeatureCollection."""
        return self.feature_collection
