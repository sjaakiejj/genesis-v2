"""
Tectonic Engine - Core Logic Module
Uses scipy for geometry and pygplates for data structures.
"""

import numpy as np
from scipy.spatial import SphericalVoronoi
import shapely
import shapely.affinity
import pygplates
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict, Any
from enum import Enum, auto
import random
import colorsys
import math
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import clip_by_rect, unary_union


class CrustType(Enum):
    OCEANIC = auto()
    CONTINENTAL = auto()


@dataclass
class PlateData:
    """Lightweight data class for plate visualization."""

    plate_id: int
    seed_point: np.ndarray  # 3D cartesian
    vertices: np.ndarray  # Nx3 array of 3D vertices
    color: Tuple[float, float, float]
    boundary_polygon: Optional[Polygon] = None
    neighbors: List[int] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    velocity_vector: Optional[np.ndarray] = None
    rotation_pole: Optional["pygplates.FiniteRotation"] = None
    crust_type: Optional[CrustType] = None
    feature: Optional["pygplates.Feature"] = None


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

        # Kinematics
        self.rotation_model: Optional[pygplates.RotationModel] = None

        # Selection state
        self._selected_ids: Set[int] = set()

        # Neighbor map (plate_id -> set of neighbor plate_ids)
        self._neighbors: Dict[int, Set[int]] = {}

        # Underlying Voronoi tessellation
        self._voronoi: SphericalVoronoi = None

        # Map from plate_id to PlateData
        self._plate_map: Dict[int, PlateData] = {}

    def generate(self, num_plates: Optional[int] = None) -> List[PlateData]:
        """
        Generate tectonic plates.

        1. Generate seed points on sphere
        2. Create SphericalVoronoi tessellation
        3. Convert regions to pygplates Features
        4. Return plate data for visualization

        Args:
           num_plates: Optional override for number of plates

        Returns:
            List of PlateData for rendering
        """
        if num_plates is not None:
            self.num_plates = num_plates

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

    def assign_kinematics(self, seed: int):
        """
        Assign random crust types and movement vectors to plates.
        Calculates velocity vectors using pygplates.
        """
        print(f"Assigning kinematics with seed {seed}")
        # Use a local random instance or seed global
        rng = random.Random(seed)

        rotations = []

        for plate in self.plates:
            # 1. Assign Crust Type (simple random for now)
            # 30% Continental, 70% Oceanic
            if rng.random() < 0.3:
                plate.crust_type = "Continental"
            else:
                plate.crust_type = "Oceanic"

            # 2. Generate Random Euler Pole & Angle
            # Random point on sphere for pole
            pole_lat = rng.uniform(-90, 90)
            pole_lon = rng.uniform(-180, 180)

            # Random angular velocity (degrees per million years)
            # Typical earth plates move ~1-10 cm/yr.
            # 1 deg/Myr is roughly 11 cm/yr at equator.
            angle_per_myr = rng.uniform(0.2, 1.2)

            finite_rotation = pygplates.FiniteRotation(
                pygplates.PointOnSphere(pole_lat, pole_lon),
                math.radians(angle_per_myr),  # pygplates uses radians for angle
            )
            plate.rotation_pole = finite_rotation

            # Create a Total Reconstruction Sequence Feature
            # We define movement of plate_id relative to anchor 9999
            # Sequence: Time 0 (Identity) -> Time 1 (Rotated)

            # Identity at 0Ma
            rot0 = pygplates.FiniteRotation.create_identity_rotation()
            prop0 = pygplates.GpmlFiniteRotation(rot0)
            sample0 = pygplates.GpmlTimeSample(prop0, 0.0)

            # Rotation at 10Ma (10 * rate)
            # Actually finite_rotation calculated above was for 1Myr "angle_per_myr"
            # So let's define the position at 1Ma
            rot1 = finite_rotation
            prop1 = pygplates.GpmlFiniteRotation(rot1)
            sample1 = pygplates.GpmlTimeSample(prop1, 1.0)

            sampling = pygplates.GpmlIrregularSampling([sample0, sample1])

            feature = pygplates.Feature.create_total_reconstruction_sequence(
                9999, plate.plate_id, sampling  # Anchor plate (virtual deep mantle)
            )

            rotations.append(feature)

        # Create Rotation Model
        self.rotation_model = pygplates.RotationModel(rotations)

        # 3. Calculate Velocity Vectors at Centroids
        for plate in self.plates:
            if plate.centroid is None:
                continue

            centroid_lat_lon = self._cartesian_to_lat_lon(plate.centroid)
            point = pygplates.PointOnSphere(centroid_lat_lon[0], centroid_lat_lon[1])

            # Calculate velocity
            # velocity is returned as a VelocityVector
            # We want velocity at 0Ma, representing motion over 1Myr
            # so we calculate rotation from 1Ma to 0Ma relative to anchor 9999
            try:
                stage_rotation = self.rotation_model.get_rotation(
                    0.0, plate.plate_id, 1.0, 9999
                )
            except pygplates.InformationModelError:
                # Fallback or skip if not connected (shouldn't happen with our logic)
                continue

            velocity_vectors = pygplates.calculate_velocities(
                [point],
                stage_rotation,
                1.0,  # time_interval_in_my
                pygplates.VelocityUnits.cms_per_yr,
            )

            if velocity_vectors:
                v_vec = velocity_vectors[0].to_xyz()  # 3D vector (x,y,z)
                plate.velocity_vector = np.array(v_vec)

    def assign_crust_types(self):
        """
        Classify plates as Oceanic or Continental based on area.
        The largest plates covering ~70% of the surface are Oceanic.
        The rest are Continental.
        """
        if not self.plates:
            return

        # 1. Calculate Areas
        plate_areas = []
        total_area = 0.0

        for plate in self.plates:
            area = 0.0
            if plate.boundary_polygon:
                area = self._calculate_plate_area_approx(plate.boundary_polygon)
            plate_areas.append((plate, area))
            total_area += area

        # 2. Sort by Area (Descending)
        plate_areas.sort(key=lambda x: x[1], reverse=True)

        # 3. Assign Types
        current_area = 0.0
        oceanic_threshold = total_area * 0.70

        for plate, area in plate_areas:
            # Create features
            # We need to convert shapely polygon to pygplates geometry
            # For now simplified: skipping geometry creation or using placeholder
            # Ideally we convert shapely -> lat/lon list -> pygplates.PolygonOnSphere

            if current_area < oceanic_threshold:
                plate.crust_type = CrustType.OCEANIC
                # In pygplates, we would create a feature of type gpml:OceanicCrust
            else:
                plate.crust_type = CrustType.CONTINENTAL
                # gpml:ContinentalCrust

            current_area += area

        print(
            f"Assigned Crust Types: {sum(1 for p in self.plates if p.crust_type == CrustType.OCEANIC)} Oceanic, {sum(1 for p in self.plates if p.crust_type == CrustType.CONTINENTAL)} Continental"
        )

    def _calculate_plate_area_approx(self, polygon: Polygon) -> float:
        """
        Calculate approximate area of a plate polygon on a sphere.
        Using simple spherical excess or just 2D area (since checking relative size).
        For sorting purposes, just summing triangle areas on unit sphere is fine.
        """
        # Simplest approximation: Area of 2D polygon in lat/lon space,
        # weighted by cos(lat) to account for spherical distortion.
        # Or even simpler: Shapely area.
        # But shapely area is in degrees^2, which distorts heavily near poles.
        # Let's use a slightly better approximation.

        return polygon.area  # Basic 2D area for V1

    def _create_pygplates_feature(self, plate: PlateData):
        """Create pygplates feature with geometry."""
        # Convert shapely polygon to pygplates geometry
        # TODO: Implement conversion if needed for GPlates export
        pass

    def _cartesian_to_lat_lon(self, v: np.ndarray) -> Tuple[float, float]:
        v_norm = v / np.linalg.norm(v)
        lat = np.degrees(np.arcsin(v_norm[2]))
        lon = np.degrees(np.arctan2(v_norm[1], v_norm[0]))
        return (lat, lon)

    def apply_boundary_noise(self, seed: int):
        """
        Apply deterministic 3D noise to plate boundaries to create curvature.
        Re-generates all plate polygons with noise.

        Args:
            seed: Random seed for noise generation
        """
        print(f"Applying boundary noise with seed {seed}")

        for plate in self.plates:
            # Check if this is a merged plate by comparing vertices count with
            # what we'd expect from a single Voronoi region.
            # Merged plates have combined vertices from multiple regions,
            # and their boundary_polygon is already a proper merged shape.
            # For merged plates, we apply noise to the existing polygon boundary.

            is_merged_plate = self._is_merged_plate(plate)

            if is_merged_plate and plate.boundary_polygon:
                # Apply noise to existing polygon boundary
                plate.boundary_polygon = self._apply_noise_to_polygon(
                    plate.boundary_polygon, seed
                )
            else:
                # Re-create polygon from original Voronoi vertices with noise
                plate.boundary_polygon = self._create_polygon_from_vertices(
                    plate.vertices, noise_seed=seed
                )

    def _is_merged_plate(self, plate: PlateData) -> bool:
        """
        Determine if a plate is a merged plate.
        Merged plates have vertices that are a concatenation of multiple regions,
        not an ordered ring of Voronoi vertices.
        """
        # A merged plate's seed_point is set to its centroid (approximately)
        # and has more vertices than typical Voronoi regions
        # Simple heuristic: if seed_point equals centroid, it's likely merged
        if plate.seed_point is not None and plate.centroid is not None:
            diff = np.linalg.norm(plate.seed_point - plate.centroid)
            if diff < 0.01:  # Very close - likely merged
                return True
        return False

    def _apply_noise_to_polygon(
        self, polygon: MultiPolygon, noise_seed: int
    ) -> MultiPolygon:
        """
        Apply noise to an existing MultiPolygon by perturbing its boundary coordinates.
        """
        result_polys = []

        for poly in polygon.geoms:
            # Get exterior coordinates (lon, lat)
            coords = list(poly.exterior.coords)
            noisy_coords = []

            for i, (lon, lat) in enumerate(coords[:-1]):  # Exclude closing point
                # Convert lat/lon to 3D Cartesian
                lat_rad = np.radians(lat)
                lon_rad = np.radians(lon)
                x = np.cos(lat_rad) * np.cos(lon_rad) * self.radius
                y = np.cos(lat_rad) * np.sin(lon_rad) * self.radius
                z = np.sin(lat_rad) * self.radius

                v = np.array([x, y, z])

                # Apply noise
                v_noisy = self._apply_noise_to_vector(v, noise_seed)

                # Convert back to lat/lon
                v_norm = v_noisy / np.linalg.norm(v_noisy)
                new_lat = np.degrees(np.arcsin(v_norm[2]))
                new_lon = np.degrees(np.arctan2(v_norm[1], v_norm[0]))

                noisy_coords.append((new_lon, new_lat))

            # Close the polygon
            if noisy_coords:
                noisy_coords.append(noisy_coords[0])

            # Create new polygon
            new_poly = Polygon(noisy_coords)
            if not new_poly.is_valid:
                new_poly = new_poly.buffer(0)

            if not new_poly.is_empty:
                if isinstance(new_poly, Polygon):
                    result_polys.append(new_poly)
                elif isinstance(new_poly, MultiPolygon):
                    result_polys.extend(new_poly.geoms)

        return MultiPolygon(result_polys) if result_polys else polygon

    def _create_polygon_from_vertices(
        self, vertices: np.ndarray, noise_seed: Optional[int] = None
    ) -> MultiPolygon:
        """
        Convert 3D vertices to a Shapely MultiPolygon on equirectangular projection.
        Handles dateline wrapping.
        Optionally applies noise to the boundary.
        """
        coords = []
        num_verts = len(vertices)

        # Increase steps if we are adding noise to capture detail
        # Increase steps if we are adding noise to capture detail
        # Lower divisor = more steps.
        step_divisor = 0.5 if noise_seed is not None else 2.0

        for i in range(num_verts):
            v_start = vertices[i]
            v_end = vertices[(i + 1) % num_verts]

            v1_norm = v_start / np.linalg.norm(v_start)
            v2_norm = v_end / np.linalg.norm(v_end)
            dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            angle = np.arccos(dot)

            # More steps for noise
            steps = max(1, int(np.degrees(angle) / step_divisor))
            if noise_seed is not None:
                steps = max(steps, 20)  # Ensure minimum resolution for noise

            # Interpolate (include start, exclude end)
            for j in range(steps):
                t = j / steps
                v = self._slerp(v_start, v_end, t)

                # Apply noise if requested
                if noise_seed is not None:
                    v = self._apply_noise_to_vector(v, noise_seed)

                v_norm = v / np.linalg.norm(v)
                lat = np.degrees(np.arcsin(v_norm[2]))
                lon = np.degrees(np.arctan2(v_norm[1], v_norm[0]))
                coords.append((lon, lat))

        # Close loop
        if coords:
            coords.append(coords[0])

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
        if len(unwrapped) > 1:
            start_lon = unwrapped[0][0]
            end_lon = unwrapped[-1][0]
            net_lon = end_lon - start_lon

            if abs(net_lon) > 180.0:
                avg_lat = np.mean([p[1] for p in unwrapped])
                pole_lat = 90.0 if avg_lat > 0 else -90.0
                unwrapped.append((end_lon, pole_lat))
                unwrapped.append((start_lon, pole_lat))

        poly = Polygon(unwrapped)

        # Buffer 0 to fix self-intersections if any
        if not poly.is_valid:
            poly = poly.buffer(0)

        polys = []

        # Create copies shifted by -360, 0, +360
        for shift in [-360.0, 0.0, 360.0]:
            shifted_poly = self._shift_polygon(poly, shift)
            try:
                clipped = clip_by_rect(shifted_poly, -180.0, -90.0, 180.0, 90.0)
                if not clipped.is_empty:
                    if isinstance(clipped, Polygon):
                        polys.append(clipped)
                    elif isinstance(clipped, MultiPolygon):
                        for p in clipped.geoms:
                            polys.append(p)
            except Exception as e:
                pass

        return MultiPolygon(polys)

    def _apply_noise_to_vector(self, v: np.ndarray, seed: int) -> np.ndarray:
        """
        Apply deterministic pseudo-random 3D noise using Fractal Brownian Motion (FBM).
        Creates more natural, varying "coastline-like" boundary deformations.
        """
        x, y, z = v

        # FBM Parameters
        octaves = 4
        frequency = 3.5  # Base frequency
        amplitude = 0.08 * self.radius  # Increased base amplitude
        persistence = 0.5
        lacunarity = 2.2

        dx = 0.0
        dy = 0.0
        dz = 0.0

        # Deterministic offset based on seed
        # Using a large prime multiplier to scramble the seed
        offset = (seed * 132049) % 10000

        current_freq = frequency
        current_amp = amplitude

        for i in range(octaves):
            # Phase shifts for this octave to de-correlate axes
            bg_phase = offset + i * 13.37

            # Simple 3D sine wave combination used as pseudo-noise
            # This is deterministic and continuous
            n1 = np.sin(y * current_freq + bg_phase) * np.cos(
                z * current_freq + bg_phase
            )
            n2 = np.sin(z * current_freq + bg_phase * 1.3) * np.cos(
                x * current_freq + bg_phase * 1.3
            )
            n3 = np.sin(x * current_freq + bg_phase * 1.7) * np.cos(
                y * current_freq + bg_phase * 1.7
            )

            dx += n1 * current_amp
            dy += n2 * current_amp
            dz += n3 * current_amp

            current_freq *= lacunarity
            current_amp *= persistence

        return np.array([x + dx, y + dy, z + dz])

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

        # Create merged polygon
        polys_to_merge = []
        for pid in selected:
            p_data = self._plate_map[pid]
            if p_data.boundary_polygon:
                polys_to_merge.append(p_data.boundary_polygon)

        merged_poly = unary_union(polys_to_merge)
        if isinstance(merged_poly, Polygon):
            merged_poly = MultiPolygon([merged_poly])

        # Create merged plate with first selected ID
        new_id = selected[0]
        merged_plate = PlateData(
            plate_id=new_id,
            vertices=combined_vertices,
            color=keep_color,
            centroid=new_centroid,
            seed_point=new_centroid,  # Use centroid as proxy for seed
            boundary_polygon=merged_poly,
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
        # 1. Remove old selected plates from everyone's lists
        for pid in list(self._neighbors.keys()):
            self._neighbors[pid] -= set(selected)

        # 2. Add new merged plate to its neighbors' lists (reciprocal)
        for n_id in merged_neighbors:
            if n_id in self._neighbors:
                self._neighbors[n_id].add(new_id)

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
        v_norm = point / np.linalg.norm(point)

        # Convert to lat/lon for point-in-polygon check
        lat = np.degrees(np.arcsin(v_norm[2]))
        lon = np.degrees(np.arctan2(v_norm[1], v_norm[0]))
        pt = Point(lon, lat)

        # Check containment first (accurate)
        for plate in self.plates:
            if plate.boundary_polygon and plate.boundary_polygon.contains(pt):
                return plate.plate_id

        # Fallback to closest centroid if not in any polygon (e.g. gaps)
        closest_id = None
        min_dist = float("inf")

        for plate in self.plates:
            if plate.centroid is None:
                continue
            dot = np.dot(point, plate.centroid)
            dist = -dot
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
