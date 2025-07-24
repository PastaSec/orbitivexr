"""
OrbitiveXR Backend Service (FastAPI)

This module defines a FastAPI application exposing REST endpoints to support
the OrbitiveXR front‑end.  It uses a local SQLite database (``orbitivexr.db``)
to store campaigns and designer profiles and implements a simple
matchmaking algorithm based on weighted campaign criteria.

Endpoints
---------

* ``POST /campaigns`` – Submit a new campaign.  Accepts JSON fields
  ``budget`` (float), ``ambiance`` (str), ``platform_pref`` (str),
  ``interactivity`` (int), ``style`` (str) and ``timeline`` (str).  Returns
  the created campaign record with an auto‑generated ID and timestamp.
* ``GET /campaigns`` – List all campaigns.
* ``POST /designers`` – Register a new designer.  Accepts JSON fields
  ``name`` (str), ``rate_tier`` (float), ``scene_tags`` (list of str),
  ``export_formats`` (list of str), ``game_logic_experience`` (int),
  ``visual_metadata`` (list of str), ``availability`` (str) and
  ``performance_score`` (float).  Returns the created designer record.
* ``GET /designers`` – List all designers.
* ``POST /match`` – Match designers to a campaign.  Accepts JSON with
  ``campaign_id`` and optional ``threshold`` (defaults to 60).  Returns a
  sorted list of designers that meet or exceed the match threshold along
  with their scores.

To run the API locally:

```
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

This will serve the API on ``http://localhost:8000``.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Database utilities
# -----------------------------------------------------------------------------

DATABASE_NAME = "orbitivexr.db"


def init_db() -> None:
    """Initialise the SQLite database if tables do not exist."""
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS campaigns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            budget REAL,
            ambiance TEXT,
            platform_pref TEXT,
            interactivity INTEGER,
            style TEXT,
            timeline TEXT,
            submitted_at TEXT
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS designers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            rate_tier REAL,
            scene_tags TEXT,
            export_formats TEXT,
            game_logic_experience INTEGER,
            visual_metadata TEXT,
            availability TEXT,
            performance_score REAL
        )
        """
    )
    conn.commit()
    conn.close()


def dict_factory(cursor, row) -> Dict[str, Any]:
    """Convert SQLite rows to dictionaries for JSON serialization."""
    d: Dict[str, Any] = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def get_db_connection() -> sqlite3.Connection:
    """Open a connection to the SQLite database with row factory."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = dict_factory
    return conn


def serialize_list(items: List[str]) -> str:
    return json.dumps(items) if items is not None else json.dumps([])


def deserialize_list(data: Optional[Any]) -> List[str]:
    """Deserialize a list stored as JSON or return the list if already a list.

    The database stores lists as JSON strings.  However, when using the
    in‑memory objects during matching (e.g. in tests), these values may
    already be Python lists.  This helper normalizes both cases.
    """
    if data is None:
        return []
    # If data is already a list, return it directly
    if isinstance(data, list):
        return data
    # If data is a JSON string, attempt to decode
    if isinstance(data, str):
        data = data.strip()
        if not data:
            return []
        try:
            return json.loads(data)
        except Exception:
            # Return empty list on failure
            return []
    # Fallback: return an empty list
    return []


# -----------------------------------------------------------------------------
# Pydantic schemas
# -----------------------------------------------------------------------------

class CampaignCreate(BaseModel):
    budget: float
    ambiance: str
    platform_pref: str
    interactivity: int
    style: str
    timeline: str


class CampaignOut(CampaignCreate):
    id: int
    submitted_at: str


class DesignerCreate(BaseModel):
    name: str
    rate_tier: float
    scene_tags: List[str] = Field(default_factory=list)
    export_formats: List[str] = Field(default_factory=list)
    game_logic_experience: int
    visual_metadata: List[str] = Field(default_factory=list)
    availability: str
    performance_score: float


class DesignerOut(DesignerCreate):
    id: int


class MatchRequest(BaseModel):
    campaign_id: int
    threshold: float = 60.0


class ScoredDesigner(BaseModel):
    designer: DesignerOut
    score: float


# -----------------------------------------------------------------------------
# Matching logic
# -----------------------------------------------------------------------------

def calculate_match_score(campaign: Dict[str, Any], designer: Dict[str, Any]) -> float:
    score = 0.0
    # Budget compatibility (20 %)
    if designer["rate_tier"] is not None and campaign["budget"] >= designer["rate_tier"]:
        score += 20.0
    # Ambiance match (20 %)
    scene_tags = deserialize_list(designer.get("scene_tags"))
    if campaign.get("ambiance") and campaign["ambiance"] in scene_tags:
        score += 20.0
    # Platform preference (15 %)
    export_formats = deserialize_list(designer.get("export_formats"))
    if campaign.get("platform_pref") and campaign["platform_pref"] in export_formats:
        score += 15.0
    # Interactivity level (15 %)
    if designer.get("game_logic_experience") is not None and campaign.get("interactivity") is not None:
        if designer["game_logic_experience"] >= campaign["interactivity"]:
            score += 15.0
    # Style aesthetic (15 %)
    visual_metadata = deserialize_list(designer.get("visual_metadata"))
    if campaign.get("style") and campaign["style"] in visual_metadata:
        score += 15.0
    # Timeline alignment (10 %) – simple lexicographic comparison as placeholder
    if designer.get("availability") and campaign.get("timeline"):
        if designer["availability"] <= campaign["timeline"]:
            score += 10.0
    # Past performance (5 %)
    if designer.get("performance_score"):
        score += float(designer["performance_score"]) * 5.0
    return score


# -----------------------------------------------------------------------------
# FastAPI application
# -----------------------------------------------------------------------------

app = FastAPI(title="OrbitiveXR API", version="0.1.0")


@app.on_event("startup")
def startup_event() -> None:
    # Initialise the database on startup
    init_db()


@app.post("/campaigns", response_model=CampaignOut, status_code=201)
def create_campaign(campaign: CampaignCreate):
    conn = get_db_connection()
    cur = conn.cursor()
    submitted_at = datetime.datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO campaigns (budget, ambiance, platform_pref, interactivity, style, timeline, submitted_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            campaign.budget,
            campaign.ambiance,
            campaign.platform_pref,
            campaign.interactivity,
            campaign.style,
            campaign.timeline,
            submitted_at,
        ),
    )
    conn.commit()
    campaign_id = cur.lastrowid
    cur.close()
    conn.close()
    return CampaignOut(id=campaign_id, submitted_at=submitted_at, **campaign.dict())


@app.get("/campaigns", response_model=List[CampaignOut])
def list_campaigns():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM campaigns")
    campaigns = cur.fetchall()
    cur.close()
    conn.close()
    return [
        CampaignOut(
            id=row["id"],
            budget=row["budget"],
            ambiance=row["ambiance"],
            platform_pref=row["platform_pref"],
            interactivity=row["interactivity"],
            style=row["style"],
            timeline=row["timeline"],
            submitted_at=row["submitted_at"],
        )
        for row in campaigns
    ]


@app.post("/designers", response_model=DesignerOut, status_code=201)
def create_designer(designer: DesignerCreate):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO designers (name, rate_tier, scene_tags, export_formats, game_logic_experience, visual_metadata, availability, performance_score)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            designer.name,
            designer.rate_tier,
            serialize_list(designer.scene_tags),
            serialize_list(designer.export_formats),
            designer.game_logic_experience,
            serialize_list(designer.visual_metadata),
            designer.availability,
            designer.performance_score,
        ),
    )
    conn.commit()
    designer_id = cur.lastrowid
    cur.close()
    conn.close()
    return DesignerOut(id=designer_id, **designer.dict())


@app.get("/designers", response_model=List[DesignerOut])
def list_designers():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM designers")
    designers = cur.fetchall()
    cur.close()
    conn.close()
    result: List[DesignerOut] = []
    for d in designers:
        result.append(
            DesignerOut(
                id=d["id"],
                name=d["name"],
                rate_tier=d["rate_tier"],
                scene_tags=deserialize_list(d.get("scene_tags")),
                export_formats=deserialize_list(d.get("export_formats")),
                game_logic_experience=d["game_logic_experience"],
                visual_metadata=deserialize_list(d.get("visual_metadata")),
                availability=d["availability"],
                performance_score=d.get("performance_score") or 0.0,
            )
        )
    return result


@app.post("/match", response_model=List[ScoredDesigner])
def match_designers(match_request: MatchRequest):
    conn = get_db_connection()
    cur = conn.cursor()
    # Fetch the campaign
    cur.execute("SELECT * FROM campaigns WHERE id = ?", (match_request.campaign_id,))
    campaign = cur.fetchone()
    if not campaign:
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail=f"Campaign {match_request.campaign_id} not found")
    # Fetch all designers
    cur.execute("SELECT * FROM designers")
    designers = cur.fetchall()
    cur.close()
    conn.close()

    scored_designers: List[ScoredDesigner] = []
    for d in designers:
        d["scene_tags"] = deserialize_list(d.get("scene_tags"))
        d["export_formats"] = deserialize_list(d.get("export_formats"))
        d["visual_metadata"] = deserialize_list(d.get("visual_metadata"))
        score = calculate_match_score(campaign, d)
        if score >= match_request.threshold:
            scored_designers.append(
                ScoredDesigner(
                    designer=DesignerOut(
                        id=d["id"],
                        name=d["name"],
                        rate_tier=d["rate_tier"],
                        scene_tags=d["scene_tags"],
                        export_formats=d["export_formats"],
                        game_logic_experience=d["game_logic_experience"],
                        visual_metadata=d["visual_metadata"],
                        availability=d["availability"],
                        performance_score=d.get("performance_score") or 0.0,
                    ),
                    score=score,
                )
            )
    # Sort descending
    scored_designers.sort(key=lambda sd: sd.score, reverse=True)
    return scored_designers
