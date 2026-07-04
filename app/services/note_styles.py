from app.core.config import Settings, ToneDefinition
from app.schemas.note import NoteStyleConfig


class NoteStyleService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def resolve(
        self,
        style: NoteStyleConfig | None,
        fallback_name: str,
        fallback_instructions: str | None,
    ) -> ToneDefinition:
        style_name = style.style_name if style else fallback_name
        custom_instructions = style.custom_instructions if style else fallback_instructions
        if style_name == "custom":
            persona = custom_instructions or "Write clean, well-structured markdown."
            return ToneDefinition(
                system_template=(
                    f"You produce markdown notes following these instructions: {persona}"
                ),
                temperature=0.5,
                top_p=0.9,
            )
        style_definition = self.settings.note_styles.get(style_name)
        if style_definition is None:
            # Style may have been renamed or edited directly in the DB; fall back gracefully.
            style_definition = self.settings.note_styles["default"]
        return style_definition
