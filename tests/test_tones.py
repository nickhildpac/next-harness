from app.core.config import Settings
from app.schemas.conversation import ToneConfig
from app.services.tones import ToneService


def test_custom_persona_is_sanitized():
    tone = ToneConfig(tone_name="custom", custom_persona="  Be {kind} \n and direct. ")

    assert tone.custom_persona == "Be kind and direct."


def test_resolves_predefined_tone():
    service = ToneService(Settings())

    resolved = service.resolve(ToneConfig(tone_name="technical"), "friendly", None)

    assert "technical" in resolved.system_template.lower()
    assert resolved.temperature <= 0.5

