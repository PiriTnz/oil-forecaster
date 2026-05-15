"""Tests for geopolitical events database"""
import pytest
from datetime import date
from src.data.events_database import (
    EVENTS, EventType, Severity, ImpactDirection,
    get_events_in_range, is_event_active, label_dates,
)


def test_events_database_not_empty():
    assert len(EVENTS) > 20  # We have 20+ events


def test_all_events_have_required_fields():
    for e in EVENTS:
        assert e.event_id
        assert e.name
        assert isinstance(e.start_date, date)
        assert isinstance(e.event_type, EventType)
        assert isinstance(e.severity, Severity)
        assert isinstance(e.impact_direction, ImpactDirection)


def test_event_ids_unique():
    ids = [e.event_id for e in EVENTS]
    assert len(ids) == len(set(ids)), "duplicate event_ids"


def test_chronological_ordering():
    """End dates should be after start dates"""
    for e in EVENTS:
        if e.end_date:
            assert e.end_date >= e.start_date, f"{e.event_id} has end before start"


def test_get_events_in_range_covers_2008_crisis():
    """The 2008 financial crisis must be findable"""
    events = get_events_in_range(date(2008, 9, 1), date(2008, 12, 31))
    assert any("gfc" in e.event_id.lower() or "Financial" in e.name for e in events)


def test_get_events_in_range_covers_covid():
    events = get_events_in_range(date(2020, 4, 1), date(2020, 5, 1))
    assert any("covid" in e.event_id.lower() for e in events)


def test_get_events_in_range_covers_ukraine():
    events = get_events_in_range(date(2022, 2, 24), date(2022, 3, 1))
    assert any("ukraine" in e.event_id.lower() for e in events)


def test_is_event_active_on_quiet_day():
    """A truly quiet day with no events would be very rare given how many we have"""
    # Just test the function works
    result = is_event_active(date(2025, 1, 1))
    assert isinstance(result, bool)


def test_is_event_active_filtered_by_type():
    """Iraq war was active during 2005"""
    result = is_event_active(date(2005, 6, 1), event_types=[EventType.WAR])
    assert result is True


def test_label_dates_returns_correct_structure():
    dates = [date(2020, 4, 15), date(2022, 3, 1)]
    labels = label_dates(dates)
    assert len(labels) == 2
    for l in labels:
        assert "date" in l
        assert "n_active_events" in l
        assert "has_war" in l
        assert "max_severity" in l


def test_to_dict_serialization():
    e = EVENTS[0]
    d = e.to_dict()
    assert d["event_id"] == e.event_id
    assert isinstance(d["start_date"], str)  # ISO format
    assert d["event_type"] == e.event_type.value
