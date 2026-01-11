Based on the video, here is the exact step-by-step workflow the presenter uses to create a realistic fantasy map using plate tectonics.

**Software Tools Required:**

- **Adobe Photoshop** (or GIMP as a free alternative)
- **GPlates** (Free open-source software)
- **G.Projector** (Free cross-platform application)

### Phase 1: Preparation and 2D Sketching

**Tool:** Photoshop

1. **Define the Concept:** Start with a written brief to avoid recreating Earth. (Example used: A Pangea-like supercontinent in the northern hemisphere beginning to break apart).
2. **Create the Document:** Create a new document with a **2:1 aspect ratio**.
3. **Set the Grid:** Overlay a **12 by 6 grid** on the canvas.
4. **Draw Plate Boundaries:** Sketch the tectonic plate boundaries as lines.
   - _Constraint:_ Ensure the left and right edges match so the map wraps around a sphere correctly.
   - _Constraint:_ Avoid "crosses" where four plates meet; ensure interactions form "T" shapes or single lines.
   - **End Result:** An equirectangular 2D line map of tectonic plates.

### Phase 2: 3D Visualization and Plate Data

**Tool:** GPlates

5. **3D Inspection:** Import the 2D map into GPlates to view it on a 3D globe.
6. **Correction:** Check the poles for distortion (squishing/elongating) and edit the lines directly in GPlates to fix them. Delete any unnecessary micro-plates.
7. **Classify Crust Types:** Mark which plates are **Continental** (thick, low density) and which are **Oceanic** (thin, high density).
   - _Method:_ Place a dot on continental plates; leave oceanic plates blank.
   - _Tip:_ Make the largest plates oceanic.
8. **Define Plate Movement:** Draw arrows to indicate the direction the plates are moving.
   - _Method:_ Locate the "mid-ocean ridge" (where plates separate) and draw arrows moving away from it. Resolve conflicting directions by assuming rotational movement.
9. **Draw Continents:** Sketch the actual landmasses inside the plates marked as "Continental."
   - _Note:_ This is "function over form"—artistic ability is not the priority here.
   - **End Result:** A 3D globe showing crust types, movement vectors, and basic continent outlines.

### Phase 3: Feature Mapping

**Tool:** GPlates (Conceptual work using the Reference Table)

10. **Analyze Boundaries:** Using the movement arrows and crust types, determine what landforms exist at the boundaries based on the reference table created at the start of the video:
    - _Convergent (colliding):_ Mountains, trenches, volcanoes.
    - _Divergent (separating):_ Ridges, rift valleys.
    - _Transform (sliding):_ Earthquakes (no major landforms).
11. **Add Islands:** Draw island chains directly into GPlates.
    - _Volcanic Arcs:_ Placed at subduction zones.
    - _Hotspots:_ Placed in the middle of plates (like Hawaii).

### Phase 4: Final Rendering and Projection

**Tool:** Photoshop

12. **Create Elevation Map:** Bring the map data back into Photoshop.
13. **Paint Elevation:** Paint the map features based on the analysis in Step 10.
    - Draw mountains where plates collide.
    - Draw rift valleys where they separate.
    - Include the **Continental Shelf** (shallow sea areas around landmasses).
    - **End Result:** A colored elevation map showing terrain height and depth.

**Tool:** G.Projector

14. **Final Projection:** Import the elevation map into G.Projector.
15. **Export:** Convert the map into various projections (e.g., heart-shaped, Robinson, etc.) to see how the world looks on a flat surface.

Based on the new video, here is the extension to the workflow for applying the **Köppen Climate Classification** system to the map.

**Software Tools Required:**

- **Adobe Photoshop** (or GIMP) - The presenter uses an image editor to paint climate zones onto the previously generated map.

**Prerequisites:**
Before starting this phase, you must have established:

- The map (created in the previous phases).
- Ocean currents.
- Prevailing wind patterns.

### Phase 5: Climate Preparation

**Tool:** Photoshop

16. **Define Variables:** Analyze your map to determine areas of high and low pressure and precipitation based on the following rules:
    - **High Precipitation:** Low-pressure areas (ITCZ, Polar Front), onshore winds, warm currents, and the **Windward** side of mountains (Orthographic Lift).
    - **Low Precipitation:** High-pressure areas (Subtropical ridges), offshore winds, cold currents, continent interiors, and the **Leeward** side of mountains (Rain Shadow).
    - **Temperature:** High at the equator/interiors; low at poles/coasts/high altitudes.
17. **Mask Mountains:** Identify "tall mountains" (approx. above 800m). Mark these off to be ignored for now, as altitude climate variation is handled separately later.

### Phase 6: Placing Tropical Climates (The Hadley Cell)

**Tool:** Photoshop
_The presenter paints these zones using specific colors in layers over the map._

18. **Tropical Rainforest (Af):**

    - **Placement:** Paint low-lying areas between **0° and 10° North/South**.
    - **Adjustments:**
      - Normally centered in continent interiors.
      - **Shift** the zone based on **Rain Shadows**: If mountains block the wind, move the rainforest to the windward side and remove it from the leeward side.
      - Reduce areas near cold currents; extend areas near warm currents.

19. **Tropical Savanna (Aw):**

    - **Placement:** Paint the regions flanking the rainforests, roughly **5° to 15-20° North/South**.
    - **Adjustments:**
      - Extend the zone further poleward (up to **25-30°**) along coasts affected by **warm currents**.
      - For thin landmasses or peninsulas, the savanna can cover the entire width of the land.

20. **Hot Deserts (BWh):**

    - **Placement:** Paint low-lying continental interiors in the poleward half of the Hadley Cell, roughly **10° to 30° North/South**.
    - **Adjustments:**
      - Place in areas with **cold coastal currents** or **offshore winds**.
      - **Do not** place where there are warm currents or onshore winds.
      - If a rain shadow (created in step 18) blocks moisture, extend the desert zone further toward the equator than usual.
      - _Note:_ On thin continents with cold currents, these deserts will be milder and foggy.

21. **Hot Steppes (BSh):**

    - **Placement:** Paint a thin "transition band" surrounding the Hot Deserts.
    - **Location:** Roughly **10° to 35° North/South**.
    - **Logic:** Place these in areas that are dry, but not quite dry enough to be full deserts.

22. **Tropical Monsoon (Am):**

    - **Placement:** Paint thin strips along coasts roughly **5° to 20° North/South**.
    - **Criteria:** Look for areas with a large landmass at 30° N/S located off a large equatorial ocean (where wind direction reverses seasonally).
    - **Adjustments:** If specific "monsoon geography" is missing (like on the presenter's map), place these strips simply where strong onshore winds hit the coast.

### End Result of Phase 6

A map with the **tropical (Type A) and dry (Type B)** climate zones fully colored in, accounting for the specific geography, mountain rain shadows, and ocean currents of your fantasy world. The temperate and polar zones remain blank for the next step.

Based on the third video in the series, here is the continuation of the workflow. This phase focuses on filling in the **Ferrel Cells** (Temperate/Continental climates) and the **Polar Cells**.

**Software Tools Required:**

- **Adobe Photoshop** (or GIMP) - Continued usage for painting specific climate zones onto the map layers.

### Phase 7: The Ferrel Cell (Continental & Subtropical Climates)

**Tool:** Photoshop

23. **Humid Continental (Dfa/Dfb):**

    - **Initial Placement:** Paint _all_ low-lying landmasses within the Ferrel Cells (**30° to 60° North/South**) with this zone temporarily.
    - **Refinement:** Later steps will overwrite parts of this, generally pushing this zone to **40°–60° North/South**.

24. **Subarctic Continental / Taiga (Dfc/Dfd):**

    - **Placement:** Paint bands between **45° and 75° North/South**.
    - **Adjustments:**
      - Skew the zone toward the equator (further south in the northern hemisphere) in regions affected by **Cold Currents** or **Offshore Winds**.

25. **Mediterranean (Csa/Csb):**

    - **Placement:** Paint regions between **30° and 45° North/South**.
    - **Condition:** Place specifically in regions affected by **Cold Currents** (typically the western sides of continents).
    - **Worldbuilding Tip:** Locate a river valley within this zone to serve as your world's "Fertile Crescent" (ideal for agriculture/annual crops).

26. **Humid Subtropical (Cfa/Cwa):**

    - **Placement:** Paint regions between **25° and 45° North/South**.
    - **Condition:** Place specifically in regions affected by **Warm Currents** (typically the eastern sides of continents).
    - _Check:_ You should now have Mediterranean climates on one side of a continent and Humid Subtropical on the opposite side.

27. **Oceanic (Cfb/Cfc):**

    - **Placement:** Paint regions between **40° and 60° North/South**.
    - **Condition:** Place in regions affected by **Warm Currents**, usually poleward of the Mediterranean zones.
    - _Note:_ If a continent is very narrow, this zone can stretch across the entire landmass. This zone includes Temperate Rainforests.

28. **Cold Deserts (BWk):**

    - **Placement:** Paint interior regions of large continents within the Ferrel Cell.
    - **Condition:** Place in **Rain Shadows** of interior mountain ranges.
    - _Note:_ These are generally located at higher altitudes than tropical deserts.

29. **Cold Steppes (BSk):**

    - **Placement:** Paint a "transition band" surrounding the Cold Deserts.
    - **Location:** Roughly **25° to 50° North/South**.
    - **Logic:** These transition between the Cold Deserts and the warmer climates towards the equator.

### Phase 8: The Polar Cell

**Tool:** Photoshop

30. **Polar Tundra (ET):**

    - **Placement:** Paint land areas between **60° and 80° North/South**.

31. **Polar Ice Caps (EF):**

    - **Placement:** Paint everything poleward of **75° North/South**.
    - **Adjustments:** Also paint the centers of very large polar landmasses (like Antarctica).

### Phase 9: Finalization

**Tool:** Photoshop / Calculation

32. **Island Fill:** Apply the same logic from all previous steps to small islands (e.g., if an island is in a warm current at 30°, make it Humid Subtropical).
33. **Analysis:** (Optional) You can use the "Albedo" method (referenced from another video) to calculate the average temperature of the planet based on the surface area coverage of these biomes. If the temperature is too hot or cold for your setting, adjust the sizes of the zones accordingly.

**Final Result:**
A complete, scientifically plausible map of an Earth-like planet featuring detailed Tectonic Plates, Continents, Mountains, and a fully classified Köppen Climate system ready for civilization building.
