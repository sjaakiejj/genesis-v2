"""
UI Manager - Panda3D DirectGUI-based UI
Provides sidebar with controls, texture preview, and selection/merge functionality.
Refactored for clean Accordion layout.
"""

from direct.gui.DirectGui import (
    DirectFrame,
    DirectButton,
    DirectLabel,
    DirectWaitBar,
    DirectEntry,
    DGG,
)
from direct.gui.OnscreenImage import OnscreenImage
from panda3d.core import (
    Texture,
    PNMImage,
    TransparencyAttrib,
    TextNode,
    CardMaker,
    Vec4,
)
from PIL import Image
from typing import Optional, Callable


class UIManager:
    """
    Manages the application UI including sidebar and controls.
    """

    # UI Constants
    SIDEBAR_COLOR = (0.12, 0.12, 0.15, 1.0)
    SECTION_HEADER_COLOR = (0.18, 0.18, 0.22, 1.0)
    SECTION_HOVER_COLOR = (0.22, 0.22, 0.26, 1.0)
    SECTION_BG_COLOR = (0.14, 0.14, 0.17, 1.0)

    BUTTON_COLOR = (0.2, 0.4, 0.6, 1.0)
    BUTTON_HOVER = (0.25, 0.45, 0.65, 1.0)

    MERGE_BUTTON_COLOR = (0.5, 0.3, 0.6, 1.0)
    MERGE_BUTTON_HOVER = (0.55, 0.35, 0.65, 1.0)

    DISABLED_COLOR = (0.3, 0.3, 0.3, 1.0)
    TEXT_COLOR = (0.9, 0.9, 0.9, 1.0)

    # Layout Constants
    HEADER_HEIGHT = 0.06
    SECTION_SPACING = 0.015
    START_Y = 0.82

    def __init__(self, base, sidebar_ratio: float = 0.22):
        self.base = base
        self.sidebar_ratio = sidebar_ratio

        # State
        self._sections = []  # List of dicts with frame, content, height, header
        self._is_generating = False

        # UI elements
        self._sidebar_bg = None
        self._elements = []
        self._texture_preview: OnscreenImage = None
        self._reference_preview: OnscreenImage = None

        # Callbacks
        self._on_generate: Optional[Callable] = None
        self._on_merge: Optional[Callable] = None
        self._on_apply_noise: Optional[Callable] = None

        # Create UI
        self._create_sidebar()

    def _create_sidebar(self):
        """Create the sidebar UI panel."""
        aspect = self.base.getAspectRatio()

        sidebar_width = self.sidebar_ratio * 2 * aspect
        right_edge = aspect
        sidebar_left = right_edge - sidebar_width
        self.sidebar_center = sidebar_left + sidebar_width / 2

        # Background
        cm = CardMaker("sidebar_bg")
        cm.setFrame(sidebar_left, right_edge + 2.0, -1, 1)
        self._sidebar_bg = self.base.aspect2d.attachNewNode(cm.generate())
        self._sidebar_bg.setColor(*self.SIDEBAR_COLOR)
        self._sidebar_bg.setBin("background", 0)

        scale = min(1.0, sidebar_width / 0.5)
        self.ui_scale = scale
        self.btn_width = 0.18 * scale

        # --- TITLE ---
        title = DirectLabel(
            text="Genesis-4",
            text_scale=0.06 * scale,
            text_fg=self.TEXT_COLOR,
            text_align=TextNode.ACenter,
            pos=(self.sidebar_center, 0, 0.92),
            parent=self.base.aspect2d,
            frameColor=(0, 0, 0, 0),
        )
        self._elements.append(title)

        # --- SECTIONS ---

        # 1. Generation Section
        s1_frame, s1_content = self._create_section("Step 1: Generation")

        # Num Plates Row
        y_cursor = -0.02
        self._create_label(s1_content, "Num Plates:", (-0.14 * scale, y_cursor))
        self._num_plates_entry = self._create_entry(
            s1_content, "12", (0.04 * scale, y_cursor)
        )

        y_cursor -= 0.07

        # Generate Button
        self._generate_button = self._create_button(
            s1_content, "Generate Plates", (0, y_cursor), self._on_generate_clicked
        )

        y_cursor -= 0.08

        # Merge Button
        self._merge_button = self._create_button(
            s1_content,
            "Merge Selected",
            (0, y_cursor),
            self._on_merge_clicked,
            color=self.MERGE_BUTTON_COLOR,
            hover=self.MERGE_BUTTON_HOVER,
        )
        self._merge_button.hide()

        # Register Section 1
        s1_height = 0.25  # Approximate expanded height
        self._add_section_to_layout(s1_frame, s1_content, s1_height)

        # 2. Morphology Section
        s2_frame, s2_content = self._create_section("Step 2: Morphology")

        y_cursor = -0.02
        self._create_label(s2_content, "Seed:", (-0.14 * scale, y_cursor))
        self._seed_entry = self._create_entry(
            s2_content, "12345", (0.04 * scale, y_cursor)
        )

        y_cursor -= 0.07

        self._apply_noise_button = self._create_button(
            s2_content, "Apply Curvature", (0, y_cursor), self._on_apply_noise_clicked
        )

        # Register Section 2
        s2_height = 0.18
        self._add_section_to_layout(s2_frame, s2_content, s2_height)

        # --- BOTTOM PANELS ---
        # Fixed area at bottom for status/map

        # Status
        self._status_label = DirectLabel(
            text="Ready",
            text_scale=0.03 * scale,
            text_fg=(0.7, 0.7, 0.7, 1),
            text_align=TextNode.ACenter,
            pos=(self.sidebar_center, 0, 0.15),
            parent=self.base.aspect2d,
            frameColor=(0, 0, 0, 0),
        )
        self._elements.append(self._status_label)

        # Selection Info
        self._selection_label = DirectLabel(
            text="",
            text_scale=0.03 * scale,
            text_fg=self.BUTTON_COLOR,
            text_align=TextNode.ACenter,
            pos=(self.sidebar_center, 0, 0.10),
            parent=self.base.aspect2d,
            frameColor=(0, 0, 0, 0),
        )
        self._elements.append(self._selection_label)

        # Progress Bar
        self._progress_bar = DirectWaitBar(
            text="",
            value=0,
            range=100,
            frameSize=(-self.btn_width, self.btn_width, -0.005, 0.005),
            frameColor=(0.1, 0.1, 0.1, 1),
            barColor=self.BUTTON_COLOR,
            pos=(self.sidebar_center, 0, 0.20),
            parent=self.base.aspect2d,
        )
        self._progress_bar.hide()
        self._elements.append(self._progress_bar)

        # Previews
        preview_w = 0.18 * scale
        preview_h = 0.09 * scale
        self._preview_scale = (preview_w, preview_h)
        self._preview_pos = -0.15
        self._ref_pos = -0.35

        # Initial layout update
        self._update_layout()

    # --- UI COMPONENT HELPERS ---

    def _create_section(self, title):
        """Create a section container."""
        width = 0.22 * self.ui_scale

        # Header Button
        header = DirectButton(
            text=title,
            text_scale=0.045 * self.ui_scale,
            text_align=TextNode.ALeft,
            text_pos=(-width + 0.02, -0.015),
            text_fg=(1, 1, 1, 1),
            frameColor=self.SECTION_HEADER_COLOR,
            frameSize=(-width, width, -0.03, 0.03),
            relief=DGG.FLAT,  # Flat style
            parent=self.base.aspect2d,
            pressEffect=0,
        )
        # Bind hover effects
        header.bind(DGG.ENTER, lambda x: header.setColor(*self.SECTION_HOVER_COLOR))
        header.bind(DGG.EXIT, lambda x: header.setColor(*self.SECTION_HEADER_COLOR))

        self._elements.append(header)

        # Content Frame
        content = DirectFrame(
            frameColor=self.SECTION_BG_COLOR,
            frameSize=(-width, width, -0.1, -0.0),  # Size updated dynamically really
            parent=self.base.aspect2d,
            relief=DGG.FLAT,
        )
        header["command"] = lambda: self._toggle_section_and_update(content)

        return header, content

    def _create_button(self, parent, text, pos, command, color=None, hover=None):
        c = color if color else self.BUTTON_COLOR
        h = hover if hover else self.BUTTON_HOVER

        btn = DirectButton(
            text=text,
            text_scale=0.04 * self.ui_scale,
            text_fg=(1, 1, 1, 1),
            frameColor=c,
            frameSize=(-self.btn_width * 0.9, self.btn_width * 0.9, -0.035, 0.045),
            relief=DGG.FLAT,
            pos=(0, 0, pos[1]),
            parent=parent,
            command=command,
        )
        # Hover
        btn.bind(DGG.ENTER, lambda x: btn.setColor(*h))
        btn.bind(DGG.EXIT, lambda x: btn.setColor(*c))
        return btn

    def _create_label(self, parent, text, pos):
        return DirectLabel(
            text=text,
            text_scale=0.035 * self.ui_scale,
            text_fg=self.TEXT_COLOR,
            text_align=TextNode.ALeft,
            pos=(pos[0], 0, pos[1]),
            parent=parent,
            frameColor=(0, 0, 0, 0),
        )

    def _create_entry(self, parent, text, pos):
        return DirectEntry(
            text=str(text),
            scale=0.035 * self.ui_scale,
            pos=(pos[0], 0, pos[1]),
            parent=parent,
            numLines=1,
            width=5,
            frameColor=(0.08, 0.08, 0.08, 1),
            text_fg=(1, 1, 1, 1),
            relief=DGG.FLAT,
        )

    # --- LAYOUT ENGINE ---

    def _add_section_to_layout(self, header, content, expanded_height):
        # Default state: Expanded
        self._sections.append(
            {
                "header": header,
                "content": content,
                "height": expanded_height,
                "expanded": True,
            }
        )

    def _toggle_section_and_update(self, content_frame):
        for section in self._sections:
            if section["content"] == content_frame:
                section["expanded"] = not section["expanded"]
                if section["expanded"]:
                    section["content"].show()
                else:
                    section["content"].hide()
                break
        self._update_layout()

    def _update_layout(self):
        """
        Recalculate positions of all sidebar elements.
        """
        current_y = self.START_Y

        width = 0.22 * self.ui_scale

        for section in self._sections:
            # Place Header
            section["header"].setPos(self.sidebar_center, 0, current_y)
            current_y -= self.HEADER_HEIGHT

            # Place Content
            if section["expanded"]:
                # Content frame top at current_y
                # Height is section["height"]
                # DirectFrame pos is typically center, but here we treat pos as top anchor
                # easier to just setPos(x, 0, current_y) and have content be negative relative to that
                section["content"].setPos(self.sidebar_center, 0, current_y)

                # Resize backdrop to look good
                h = section["height"]
                section["content"]["frameSize"] = (-width, width, -h, 0)

                current_y -= h

            current_y -= self.SECTION_SPACING

    # --- CALLBACKS & PUBLIC API ---

    def set_generate_callback(self, callback: Callable):
        self._on_generate = callback

    def set_merge_callback(self, callback: Callable):
        self._on_merge = callback

    def set_apply_noise_callback(self, callback: Callable):
        self._on_apply_noise = callback

    def _on_generate_clicked(self):
        if self._on_generate:
            try:
                num_plates = int(self._num_plates_entry.get())
                self._on_generate(num_plates)
            except ValueError:
                self.set_status("Invalid number!")

    def _on_merge_clicked(self):
        if self._on_merge:
            self._on_merge()

    def _on_apply_noise_clicked(self):
        if self._on_apply_noise:
            try:
                seed_val = int(self._seed_entry.get())
                self._on_apply_noise(seed_val)
            except ValueError:
                self.set_status("Invalid seed!")

    def show_progress(self, visible: bool = True):
        if visible:
            self._progress_bar.show()
        else:
            self._progress_bar.hide()

    def update_progress(self, percentage: float, message: str = ""):
        self._progress_bar["value"] = percentage
        if message:
            self._status_label["text"] = message

    def set_status(self, status: str):
        self._status_label["text"] = status

    def set_generating(self, is_generating: bool):
        self._is_generating = is_generating
        if is_generating:
            self._generate_button["state"] = DGG.DISABLED
            self._generate_button.setColor(*self.DISABLED_COLOR)
            self._generate_button["text"] = "Working..."
            self.show_progress(True)
        else:
            self._generate_button["state"] = DGG.NORMAL
            self._generate_button.setColor(*self.BUTTON_COLOR)
            self._generate_button["text"] = "Generate Plates"
            self.show_progress(False)

    def update_selection_state(self, count: int, can_merge: bool):
        if count == 0:
            self._selection_label["text"] = ""
        else:
            self._selection_label["text"] = f"{count} Selected"

        if can_merge and count >= 2:
            self._merge_button.show()
        else:
            self._merge_button.hide()

    def update_texture_preview(self, image: Image.Image):
        if self._texture_preview is not None:
            self._texture_preview.destroy()

        texture = self._pil_to_texture(image)
        pw, ph = self._preview_scale

        self._texture_preview = OnscreenImage(
            image=texture,
            pos=(self.sidebar_center, 0, self._preview_pos),
            scale=(pw, 1, ph),
            parent=self.base.aspect2d,
        )
        self._texture_preview.setTransparency(TransparencyAttrib.MAlpha)

    def update_reference_preview(self, image: Image.Image):
        if self._reference_preview is not None:
            self._reference_preview.destroy()

        texture = self._pil_to_texture(image)
        pw, ph = self._preview_scale

        self._reference_preview = OnscreenImage(
            image=texture,
            pos=(self.sidebar_center, 0, self._ref_pos),
            scale=(pw, 1, ph),
            parent=self.base.aspect2d,
        )
        self._reference_preview.setTransparency(TransparencyAttrib.MAlpha)

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

        texture = Texture("preview_texture")
        texture.load(pnm)
        return texture

    def destroy(self):
        """Clean up UI elements."""
        for elem in self._elements:
            elem.destroy()
        self._elements.clear()
        if self._sidebar_bg:
            self._sidebar_bg.removeNode()
        if self._texture_preview:
            self._texture_preview.destroy()
        if self._reference_preview:
            self._reference_preview.destroy()
