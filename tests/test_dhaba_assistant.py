from src.services.dhaba_assistant import DhabaRecommendationEngine


def test_dhaba_assistant_prefers_rest_stop_for_drowsy_driver():
    engine = DhabaRecommendationEngine()
    places = [
        {
            "name": "Fuel Only Station",
            "category": "⛽ Fuel & Service Station",
            "distance_km": 1.2,
            "lat": 13.0,
            "lon": 77.0,
            "maps_url": "https://example.com/fuel",
            "amenities": {"fuel": True},
        },
        {
            "name": "Family Dhaba",
            "category": "🍲 Highway Dhaba",
            "distance_km": 1.4,
            "lat": 13.0,
            "lon": 77.0,
            "maps_url": "https://example.com/dhaba",
            "amenities": {"food": True, "restroom": True, "parking": True},
        },
        {
            "name": "Far Cafe",
            "category": "☕ Refreshment Layover",
            "distance_km": 4.8,
            "lat": 13.0,
            "lon": 77.0,
            "maps_url": "https://example.com/cafe",
            "amenities": {"food": True},
        },
    ]

    result = engine.recommend(
        places,
        driver_lat=13.0,
        driver_lon=77.0,
        context="CONFIRMED_DROWSY",
        language="hinglish",
    )

    assert result.places[0].name == "Family Dhaba"
    assert "best option" in result.summary.lower()
    assert result.speech_language_code == "en-IN"
    assert "food" in " ".join(result.places[0].reasons).lower()

