"""
Smart Dhaba recommendation engine.

Ranks nearby rest stops using a lightweight score model that prefers
short distance, restrooms, food, fuel, and parking depending on the
driver's current state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import radians, sin, cos, atan2, sqrt
from typing import Dict, Any, Iterable, List, Optional, Tuple


LANGUAGE_CODES = {
    "english": "en-IN",
    "hindi": "hi-IN",
    "hinglish": "en-IN",
}


@dataclass
class DhabaPlace:
    name: str
    category: str
    distance_km: float
    lat: float
    lon: float
    maps_url: str
    amenities: Dict[str, bool] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "distance_km": round(self.distance_km, 1),
            "lat": self.lat,
            "lon": self.lon,
            "maps_url": self.maps_url,
            "amenities": self.amenities,
            "score": round(self.score, 1),
            "reasons": self.reasons,
        }


@dataclass
class DhabaRecommendation:
    places: List[DhabaPlace]
    summary: str
    voice_prompt: str
    speech_language_code: str
    context: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "places": [place.to_dict() for place in self.places],
            "summary": self.summary,
            "voice_prompt": self.voice_prompt,
            "speech_language_code": self.speech_language_code,
            "context": self.context,
            "top_pick": self.places[0].to_dict() if self.places else None,
        }


class DhabaRecommendationEngine:
    """Score and phrase nearby highway stop recommendations."""

    def __init__(self):
        self.language_codes = LANGUAGE_CODES.copy()

    @staticmethod
    def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        d_lat = radians(lat2 - lat1)
        d_lon = radians(lon2 - lon1)
        a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
        return round(6371 * 2 * atan2(sqrt(a), sqrt(1 - a)), 1)

    def build_fallback_places(self, lat: float, lon: float) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Shree Highway Punjabi Dhaba & Tea Stall",
                "category": "🍲 Highway Dhaba",
                "distance_km": 1.2,
                "lat": lat + 0.008,
                "lon": lon + 0.005,
                "maps_url": f"https://www.google.com/maps/search/dhaba/@{lat},{lon},14z",
                "amenities": {"food": True, "restroom": True, "parking": True},
            },
            {
                "name": "Indian Oil Fuel Station & 24/7 Rest Stop",
                "category": "⛽ Fuel & Service Station",
                "distance_km": 2.1,
                "lat": lat + 0.015,
                "lon": lon + 0.012,
                "maps_url": f"https://www.google.com/maps/search/petrol+pump/@{lat},{lon},14z",
                "amenities": {"fuel": True, "restroom": True, "parking": True},
            },
            {
                "name": "Highway Grand Tea & Coffee Plaza",
                "category": "☕ Refreshment Layover",
                "distance_km": 3.4,
                "lat": lat + 0.022,
                "lon": lon + 0.018,
                "maps_url": f"https://www.google.com/maps/search/cafe/@{lat},{lon},14z",
                "amenities": {"food": True, "restroom": True, "parking": True},
            },
            {
                "name": "National Highway Toll Plaza Parking Area",
                "category": "🅿️ Safe Rest Area",
                "distance_km": 4.5,
                "lat": lat + 0.030,
                "lon": lon + 0.025,
                "maps_url": f"https://www.google.com/maps/search/rest+area/@{lat},{lon},14z",
                "amenities": {"parking": True, "restroom": True},
            },
        ]

    def _normalize_place(self, place: Dict[str, Any], driver_lat: float, driver_lon: float) -> DhabaPlace:
        lat = float(place.get("lat", driver_lat))
        lon = float(place.get("lon", driver_lon))
        distance_km = float(place.get("distance_km") or self._distance_km(driver_lat, driver_lon, lat, lon))
        category = str(place.get("category", "Rest Stop"))
        tags = place.get("tags", {}) if isinstance(place.get("tags", {}), dict) else {}
        amenities = dict(place.get("amenities", {}) or {})

        category_lower = category.lower()
        if "fuel" in category_lower or tags.get("amenity") == "fuel":
            amenities.setdefault("fuel", True)
        if "restaurant" in category_lower or "cafe" in category_lower or tags.get("amenity") in {"restaurant", "cafe"}:
            amenities.setdefault("food", True)
        if "rest" in category_lower or tags.get("highway") == "rest_area":
            amenities.setdefault("restroom", True)

        return DhabaPlace(
            name=str(place.get("name", "Highway Rest Stop")),
            category=category,
            distance_km=distance_km,
            lat=lat,
            lon=lon,
            maps_url=str(place.get("maps_url", f"https://www.google.com/maps/search/{lat},{lon}")),
            amenities=amenities,
            tags=tags,
        )

    def _context_weights(self, context: str) -> Dict[str, float]:
        ctx = context.lower()
        if ctx in {"confirmed_drowsy", "persistent_drowsy", "drowsy"}:
            return {"distance": 0.30, "food": 0.20, "restroom": 0.25, "parking": 0.10, "fuel": 0.15}
        if ctx in {"recovering", "rest"}:
            return {"distance": 0.35, "food": 0.15, "restroom": 0.20, "parking": 0.15, "fuel": 0.15}
        return {"distance": 0.45, "food": 0.10, "restroom": 0.15, "parking": 0.10, "fuel": 0.20}

    def _score_place(self, place: DhabaPlace, context: str) -> DhabaPlace:
        weights = self._context_weights(context)
        score = 0.0
        reasons: List[str] = []

        distance_bonus = max(0.0, 30.0 - (place.distance_km * 4.0))
        score += distance_bonus * weights["distance"]
        if place.distance_km <= 2.0:
            reasons.append("close by")

        if place.amenities.get("food"):
            score += 18.0 * weights["food"]
            reasons.append("food available")
        if place.amenities.get("restroom"):
            score += 18.0 * weights["restroom"]
            reasons.append("restrooms likely available")
        if place.amenities.get("parking"):
            score += 10.0 * weights["parking"]
            reasons.append("safe parking")
        if place.amenities.get("fuel"):
            score += 16.0 * weights["fuel"]
            reasons.append("fuel available")

        if "24/7" in place.name or "open" in place.tags.get("opening_hours", "").lower():
            score += 5.0
            reasons.append("likely open now")

        if "dhaba" in place.category.lower() or "restaurant" in place.category.lower():
            score += 5.0

        place.score = round(score, 1)
        place.reasons = reasons or ["reasonable rest stop"]
        return place

    def _voice_language(self, language: str) -> str:
        return self.language_codes.get(language.lower(), "en-IN")

    def _format_reasons(self, reasons: Iterable[str], language: str) -> str:
        items = list(dict.fromkeys(reasons))
        if not items:
            items = ["it is a practical stop"]

        if language == "hindi":
            mapping = {
                "close by": "yeh kareeb hai",
                "food available": "khana mil jayega",
                "restrooms likely available": "washroom ka chance hai",
                "safe parking": "safe parking hai",
                "fuel available": "fuel available hai",
                "likely open now": "abhi open hone ki sambhavna hai",
                "reasonable rest stop": "yeh achha rest stop lag raha hai",
            }
            translated = [mapping.get(item, item) for item in items[:3]]
            return ", ".join(translated)

        if language == "hinglish":
            mapping = {
                "close by": "close hai",
                "food available": "food mil jayega",
                "restrooms likely available": "restroom mil sakte hain",
                "safe parking": "parking safe hai",
                "fuel available": "fuel available hai",
                "likely open now": "open hone ka chance hai",
                "reasonable rest stop": "yeh practical stop hai",
            }
            translated = [mapping.get(item, item) for item in items[:3]]
            return ", ".join(translated)

        return ", ".join(items[:3])

    def _build_summary(self, places: List[DhabaPlace], language: str) -> Tuple[str, str]:
        if not places:
            if language == "hindi":
                return (
                    "Abhi koi suitable rest stop nahi mila.",
                    "Mujhe nearby koi suitable dhaba nahi mila. Main fallback options dikhane wala hoon."
                )
            if language == "hinglish":
                return (
                    "Abhi koi suitable rest stop nahi mila.",
                    "Mujhe nearby koi suitable dhaba nahi mila. Main fallback options dikha raha hoon."
                )
            return (
                "No suitable rest stop found right now.",
                "I could not find a nearby rest stop, so I am showing the safest fallback options."
            )

        top = places[0]
        reason_text = self._format_reasons(top.reasons, language)
        distance_text = f"{top.distance_km:.1f} km"
        if language == "hindi":
            summary = f"Maine {len(places)} nearby stop dekhe. Sabse behtar {top.name} hai, {distance_text} door, kyunki {reason_text}."
            prompt = f"Dhaba assistant active hai. {top.name} sabse accha option hai, {distance_text} door. {reason_text}."
        elif language == "hinglish":
            summary = f"Maine {len(places)} nearby stops dekhe. Best option {top.name} hai, {distance_text} door, because {reason_text}."
            prompt = f"Dhaba assistant active hai. {top.name} best option hai, {distance_text} door. {reason_text}."
        else:
            summary = f"I found {len(places)} nearby stops. Best option is {top.name}, {distance_text} away, because {reason_text}."
            prompt = f"Dhaba assistant active. {top.name} is the best option, {distance_text} away. {reason_text}."
        return summary, prompt

    def recommend(
        self,
        raw_places: Iterable[Dict[str, Any]],
        driver_lat: float,
        driver_lon: float,
        context: str = "alert",
        language: str = "english",
        limit: int = 6,
    ) -> DhabaRecommendation:
        places = [self._normalize_place(place, driver_lat, driver_lon) for place in raw_places]
        scored = [self._score_place(place, context) for place in places]
        scored.sort(key=lambda item: (-item.score, item.distance_km, item.name.lower()))
        scored = scored[:limit]
        summary, prompt = self._build_summary(scored, language)
        return DhabaRecommendation(
            places=scored,
            summary=summary,
            voice_prompt=prompt,
            speech_language_code=self._voice_language(language),
            context=context,
        )

