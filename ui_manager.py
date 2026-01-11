"""
UI Manager - Panda3D DirectGUI-based UI
Provides sidebar with controls, texture preview, and selection/merge functionality.
"""

from direct.gui.DirectGui import (
    DirectFrame,
    DirectButton,
    DirectLabel,
    DirectWaitBar,
)
from direct.gui.OnscreenImage import OnscreenImage
from panda3d.core import (
    Texture,
    PNMImage,
    TransparencyAttrib,
    TextNode,
    CardMaker,
)
from PIL import Image
from typing import Optional, Callable


class UIManager:
    """
    Manages the application UI including sidebar and controls.
    """

    # UI Constants
    SIDEBAR_COLOR = (0.12, 0.12, 0.15, 1.0)
    BUTTON_COLOR = (0.25, 0.45, 0.65, 1.0)
    MERGE_BUTTON_COLOR = (0.55, 0.35, 0.65, 1.0)  # Purple for merge
    DISABLED_COLOR = (0.3, 0.3, 0.3, 1.0)
    TEXT_COLOR = (0.9, 0.9, 0.9, 1.0)

    def __init__(self, base, sidebar_ratio: float = 0.22):
        """
        Initialize the UI manager.

        Args:
            base: The ShowBase instance
            sidebar_ratio: Fraction of window width for sidebar (0-1)
        """
        self.base = base
        self.sidebar_ratio = sidebar_ratio

        # UI elements
        self._sidebar_bg = None
        self._elements = []
        self._texture_preview: OnscreenImage = None
        self._reference_preview: OnscreenImage = None

        # Callbacks
        self._on_generate: Optional[Callable] = None
        self._on_merge: Optional[Callable] = None

        # Create UI
        self._create_sidebar()

    def _create_sidebar(self):
        """Create the sidebar UI panel."""
        aspect = self.base.getAspectRatio()

        sidebar_width = self.sidebar_ratio * 2 * aspect
        right_edge = aspect
        sidebar_left = right_edge - sidebar_width
        sidebar_center = sidebar_left + sidebar_width / 2

        # Background
        cm = CardMaker("sidebar_bg")
        cm.setFrame(sidebar_left, right_edge + 2.0, -1, 1)
        self._sidebar_bg = self.base.aspect2d.attachNewNode(cm.generate())
        self._sidebar_bg.setColor(*self.SIDEBAR_COLOR)
        self._sidebar_bg.setBin("background", 0)

        scale = min(1.0, sidebar_width / 0.5)
        btn_width = 0.15 * scale

        # Title
        title = DirectLabel(
            text="Tectonic Generator",
            text_scale=0.055 * scale,
            text_fg=self.TEXT_COLOR,
            text_align=TextNode.ACenter,
            pos=(sidebar_center, 0, 0.88),
            parent=self.base.aspect2d,
            frameColor=(0, 0, 0, 0),
            text_mayChange=False,
        )
        self._elements.append(title)

        # Generate button
        self._generate_button = DirectButton(
            text="Generate Plates",
            text_scale=0.04 * scale,
            text_fg=(1, 1, 1, 1),
            frameColor=self.BUTTON_COLOR,
            frameSize=(-btn_width, btn_width, -0.035, 0.045),
            pos=(sidebar_center, 0, 0.76),
            parent=self.base.aspect2d,
            command=self._on_generate_clicked,
        )
        self._elements.append(self._generate_button)

        # Progress bar
        self._progress_bar = DirectWaitBar(
            text="",
            value=0,
            range=100,
            frameSize=(-btn_width, btn_width, -0.01, 0.01),
            frameColor=(0.2, 0.2, 0.2, 1),
            barColor=(0.3, 0.7, 0.4, 1),
            pos=(sidebar_center, 0, 0.66),
            parent=self.base.aspect2d,
        )
        self._progress_bar.hide()
        self._elements.append(self._progress_bar)

        # Progress label
        self._progress_label = DirectLabel(
            text="Ready",
            text_scale=0.032 * scale,
            text_fg=self.TEXT_COLOR,
            text_align=TextNode.ACenter,
            pos=(sidebar_center, 0, 0.60),
            parent=self.base.aspect2d,
            frameColor=(0, 0, 0, 0),
        )
        self._elements.append(self._progress_label)

        # --- Selection Section ---
        selection_label = DirectLabel(
            text="Selection:",
            text_scale=0.035 * scale,
            text_fg=self.TEXT_COLOR,
            text_align=TextNode.ACenter,
            pos=(sidebar_center, 0, 0.50),
            parent=self.base.aspect2d,
            frameColor=(0, 0, 0, 0),
            text_mayChange=False,
        )
        self._elements.append(selection_label)

        # Selection count label
        self._selection_label = DirectLabel(
            text="0 plates selected",
            text_scale=0.028 * scale,
            text_fg=(0.7, 0.7, 0.7, 1),
            text_align=TextNode.ACenter,
            pos=(sidebar_center, 0, 0.44),
            parent=self.base.aspect2d,
            frameColor=(0, 0, 0, 0),
        )
        self._elements.append(self._selection_label)

        # Merge button (initially hidden)
        self._merge_button = DirectButton(
            text="Merge Selected",
            text_scale=0.035 * scale,
            text_fg=(1, 1, 1, 1),
            frameColor=self.MERGE_BUTTON_COLOR,
            frameSize=(-btn_width, btn_width, -0.03, 0.04),
            pos=(sidebar_center, 0, 0.36),
            parent=self.base.aspect2d,
            command=self._on_merge_clicked,
        )
        self._merge_button.hide()
        self._elements.append(self._merge_button)

        # --- Plate Map Section ---
        preview_label = DirectLabel(
            text="Plate Map:",
            text_scale=0.035 * scale,
            text_fg=self.TEXT_COLOR,
            text_align=TextNode.ACenter,
            pos=(sidebar_center, 0, 0.26),
            parent=self.base.aspect2d,
            frameColor=(0, 0, 0, 0),
            text_mayChange=False,
        )
        self._elements.append(preview_label)

        preview_w = 0.15 * scale
        preview_h = 0.075 * scale
        self._preview_frame = DirectFrame(
            frameColor=(0.08, 0.08, 0.08, 1),
            frameSize=(-preview_w, preview_w, -preview_h, preview_h),
            pos=(sidebar_center, 0, 0.10),
            parent=self.base.aspect2d,
        )
        self._elements.append(self._preview_frame)

        # --- Reference Map Section ---
        ref_label = DirectLabel(
            text="Voronoi Ref:",
            text_scale=0.035 * scale,
            text_fg=self.TEXT_COLOR,
            text_align=TextNode.ACenter,
            pos=(sidebar_center, 0, -0.02),
            parent=self.base.aspect2d,
            frameColor=(0, 0, 0, 0),
            text_mayChange=False,
        )
        self._elements.append(ref_label)

        self._ref_frame = DirectFrame(
            frameColor=(0.08, 0.08, 0.08, 1),
            frameSize=(-preview_w, preview_w, -preview_h, preview_h),
            pos=(sidebar_center, 0, -0.16),
            parent=self.base.aspect2d,
        )
        self._elements.append(self._ref_frame)

        self._sidebar_center = sidebar_center
        self._preview_scale = (preview_w, preview_h)
        self._preview_pos = 0.10
        self._ref_pos = -0.16

        # Status label
        self._status_label = DirectLabel(
            text="Right-click plates to select",
            text_scale=0.026 * scale,
            text_fg=(0.6, 0.6, 0.6, 1),
            text_align=TextNode.ACenter,
            pos=(sidebar_center, 0, -0.28),
            parent=self.base.aspect2d,
            frameColor=(0, 0, 0, 0),
        )
        self._elements.append(self._status_label)

        # Instructions
        instructions = DirectLabel(
            text="Controls:\n• Drag to rotate\n• Scroll to zoom\n• Space to generate\n• Right-click to select",
            text_scale=0.022 * scale,
            text_fg=(0.45, 0.45, 0.45, 1),
            text_align=TextNode.ACenter,
            pos=(sidebar_center, 0, -0.45),
            parent=self.base.aspect2d,
            frameColor=(0, 0, 0, 0),
            text_mayChange=False,
        )
        self._elements.append(instructions)

    def set_generate_callback(self, callback: Callable):
        """Set the callback for the generate button."""
        self._on_generate = callback

    def set_merge_callback(self, callback: Callable):
        """Set the callback for the merge button."""
        self._on_merge = callback

    def _on_generate_clicked(self):
        """Handle generate button click."""
        if self._on_generate:
            self._on_generate()

    def _on_merge_clicked(self):
        """Handle merge button click."""
        if self._on_merge:
            self._on_merge()

    def show_progress(self, visible: bool = True):
        """Show or hide the progress bar."""
        if visible:
            self._progress_bar.show()
        else:
            self._progress_bar.hide()

    def update_progress(self, percentage: float, message: str = ""):
        """Update the progress bar and label."""
        self._progress_bar["value"] = percentage
        if message:
            self._progress_label["text"] = message

    def set_status(self, status: str):
        """Set the status label text."""
        self._status_label["text"] = status

    def set_generating(self, is_generating: bool):
        """Update UI state during generation."""
        if is_generating:
            self._generate_button["state"] = 0
            self._generate_button["text"] = "Generating..."
            self._generate_button["frameColor"] = self.DISABLED_COLOR
            self.show_progress(True)
        else:
            self._generate_button["state"] = 1
            self._generate_button["text"] = "Generate Plates"
            self._generate_button["frameColor"] = self.BUTTON_COLOR
            self.show_progress(False)

    def update_selection_state(self, count: int, can_merge: bool):
        """
        Update selection display and merge button state.

        Args:
            count: Number of selected plates
            can_merge: Whether selected plates can be merged
        """
        # Update selection label
        if count == 0:
            self._selection_label["text"] = "0 plates selected"
        elif count == 1:
            self._selection_label["text"] = "1 plate selected"
        else:
            self._selection_label["text"] = f"{count} plates selected"

        # Update merge button
        if can_merge and count >= 2:
            self._merge_button.show()
            self._merge_button["state"] = 1
            self._merge_button["frameColor"] = self.MERGE_BUTTON_COLOR
        else:
            self._merge_button.hide()

    def update_texture_preview(self, image: Image.Image):
        """Update the texture preview with a PIL image."""
        if self._texture_preview is not None:
            self._texture_preview.destroy()

        texture = self._pil_to_texture(image)
        pw, ph = self._preview_scale

        self._texture_preview = OnscreenImage(
            image=texture,
            pos=(self._sidebar_center, 0, self._preview_pos),
            scale=(pw, 1, ph),
            parent=self.base.aspect2d,
        )
        self._texture_preview.setTransparency(TransparencyAttrib.MAlpha)

    def update_reference_preview(self, image: Image.Image):
        """Update the reference Voronoi texture preview."""
        if self._reference_preview is not None:
            self._reference_preview.destroy()

        texture = self._pil_to_texture(image)
        pw, ph = self._preview_scale

        self._reference_preview = OnscreenImage(
            image=texture,
            pos=(self._sidebar_center, 0, self._ref_pos),
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
