from app.core.config import Settings, ToneDefinition
from app.schemas.conversation import ToneConfig


class ToneService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def resolve(self, tone: ToneConfig | None, fallback_name: str, fallback_persona: str | None):
        tone_name = tone.tone_name if tone else fallback_name
        custom_persona = tone.custom_persona if tone else fallback_persona
        if tone_name == "custom":
            persona = custom_persona or "Be helpful and adapt to the user's requested style."
            return ToneDefinition(
                system_template=f"You are an assistant with this persona: {persona}",
                temperature=0.6,
                top_p=0.9,
            )
        tone_definition = self.settings.tones.get(tone_name)
        if tone_definition is None:
            # A tone stored before a rename (or edited directly in the DB) must not break sends.
            tone_definition = self.settings.tones["professional"]
        return tone_definition

