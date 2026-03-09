"""
database.py — SQLAlchemy models and DB utilities for STYLO.

Uses PostgreSQL in production (DATABASE_URL env var) and
SQLite locally as a fallback — zero config to get started.
"""

import os
import json
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    DateTime, Text, JSON, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool

# ── Engine setup ──────────────────────────────────────────────────────────────

DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///stylo.db')

# Render / Heroku ship postgres:// but SQLAlchemy 1.4+ needs postgresql://
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

_connect_args = {}
_kwargs = {}
if DATABASE_URL.startswith('sqlite'):
    _connect_args = {'check_same_thread': False}
    _kwargs = {'poolclass': StaticPool}

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    **_kwargs
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


# ── Models ────────────────────────────────────────────────────────────────────

class Analysis(Base):
    """One analysis request — stores inputs, outputs, and timing."""
    __tablename__ = 'analyses'

    id               = Column(Integer, primary_key=True, index=True)
    created_at       = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Input
    filename         = Column(String(255), nullable=True)
    file_size_kb     = Column(Float, nullable=True)
    target_style     = Column(String(64), nullable=False)

    # Image characteristics
    img_width        = Column(Integer, nullable=True)
    img_height       = Column(Integer, nullable=True)
    brightness       = Column(Float, nullable=True)
    contrast         = Column(Float, nullable=True)
    texture          = Column(Float, nullable=True)

    # Classifier output
    predicted_style  = Column(String(64), nullable=True)   # None if no probe loaded
    confidence       = Column(Float, nullable=True)
    all_scores       = Column(JSON, nullable=True)          # {style: prob, ...}

    # User feedback (collected after the fact)
    user_rating      = Column(Integer, nullable=True)       # 1–5
    user_comment     = Column(Text, nullable=True)

    # Timing
    processing_ms    = Column(Integer, nullable=True)

    __table_args__ = (
        Index('ix_analyses_target_style', 'target_style'),
        Index('ix_analyses_predicted_style', 'predicted_style'),
        Index('ix_analyses_created_at', 'created_at'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'filename': self.filename,
            'target_style': self.target_style,
            'predicted_style': self.predicted_style,
            'confidence': self.confidence,
            'all_scores': self.all_scores,
            'user_rating': self.user_rating,
        }


class StyleStats(Base):
    """
    Materialised daily counts per style — updated on each analysis.
    Used for dashboard analytics without expensive GROUP BY queries.
    """
    __tablename__ = 'style_stats'

    id              = Column(Integer, primary_key=True)
    style           = Column(String(64), unique=True, nullable=False)
    request_count   = Column(Integer, default=0)
    avg_confidence  = Column(Float, default=0.0)
    last_updated    = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ── Helpers ───────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables. Safe to call multiple times."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """Yield a DB session; always closed after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def log_analysis(
    db: Session,
    *,
    filename: str,
    file_size_kb: float,
    target_style: str,
    feedback: dict,
    processing_ms: int,
) -> Analysis:
    """
    Persist one analysis result and update StyleStats.
    Returns the saved Analysis row.
    """
    chars = feedback.get('technical_analysis', {}).get('image_characteristics', {})
    clf   = feedback.get('classification', {})

    row = Analysis(
        filename        = filename,
        file_size_kb    = round(file_size_kb, 2),
        target_style    = target_style,
        img_width       = chars.get('dimensions', [None, None])[0],
        img_height      = chars.get('dimensions', [None, None])[1],
        brightness      = chars.get('brightness'),
        contrast        = chars.get('contrast'),
        texture         = chars.get('texture_complexity'),
        predicted_style = clf.get('predicted_style'),
        confidence      = clf.get('confidence'),
        all_scores      = clf.get('all_scores'),
        processing_ms   = processing_ms,
    )
    db.add(row)

    # Update rolling stats for target style
    stats = db.query(StyleStats).filter_by(style=target_style).first()
    if stats is None:
        stats = StyleStats(style=target_style, request_count=0, avg_confidence=0.0)
        db.add(stats)

    n = stats.request_count
    conf = clf.get('confidence') or 0.0
    stats.avg_confidence = round((stats.avg_confidence * n + conf) / (n + 1), 4)
    stats.request_count  = n + 1
    stats.last_updated   = datetime.now(timezone.utc)

    db.commit()
    db.refresh(row)
    return row


def get_recent_analyses(db: Session, limit: int = 20) -> list:
    return (
        db.query(Analysis)
        .order_by(Analysis.created_at.desc())
        .limit(limit)
        .all()
    )


def get_style_stats(db: Session) -> list:
    return db.query(StyleStats).order_by(StyleStats.request_count.desc()).all()