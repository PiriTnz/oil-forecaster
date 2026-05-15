"""
Curated database of major geopolitical and economic events affecting oil markets (2000-2025).
Each event is labeled with type, severity, region, and impact direction.
This is used for:
  1. Training regime-aware models
  2. Backtesting strategies during specific event types
  3. Validating model behavior during known crises
"""
from dataclasses import dataclass, asdict
from datetime import date
from enum import Enum
from typing import Optional
import json


class EventType(str, Enum):
    WAR = "war"                          # Active military conflict
    SANCTIONS = "sanctions"              # Economic sanctions on producers
    OPEC_ACTION = "opec_action"          # Production cuts/increases
    SUPPLY_DISRUPTION = "supply_disruption"  # Pipeline attacks, refinery fires
    FINANCIAL_CRISIS = "financial_crisis"
    PANDEMIC = "pandemic"
    DIPLOMATIC = "diplomatic"            # Major agreements/breakdowns
    NATURAL_DISASTER = "natural_disaster"  # Hurricanes affecting Gulf, etc.


class Severity(str, Enum):
    LOW = "low"          # < 5% price move expected
    MEDIUM = "medium"    # 5-15% move
    HIGH = "high"        # 15-30% move
    EXTREME = "extreme"  # > 30% move


class ImpactDirection(str, Enum):
    BULLISH = "bullish"  # Price up
    BEARISH = "bearish"  # Price down
    MIXED = "mixed"


@dataclass
class GeopoliticalEvent:
    """A labeled event affecting oil markets"""
    event_id: str
    name: str
    start_date: date
    end_date: Optional[date]  # None = ongoing or single-day
    event_type: EventType
    severity: Severity
    impact_direction: ImpactDirection
    region: str
    description: str
    affected_producers: list[str]  # Country names
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start_date"] = self.start_date.isoformat()
        d["end_date"] = self.end_date.isoformat() if self.end_date else None
        d["event_type"] = self.event_type.value
        d["severity"] = self.severity.value
        d["impact_direction"] = self.impact_direction.value
        return d


# =============================================================================
# Master event database (2000-2025)
# =============================================================================
# Note: dates are approximate based on widely-reported start of market impact.
# Sources: EIA reports, IMF analyses, academic papers on oil price shocks.
# This should be refreshed via the fetch_events pipeline using news APIs.

EVENTS: list[GeopoliticalEvent] = [
    # --- 2001-2005 ---
    GeopoliticalEvent(
        event_id="2001-09-11",
        name="September 11 attacks",
        start_date=date(2001, 9, 11),
        end_date=date(2001, 10, 15),
        event_type=EventType.WAR,
        severity=Severity.MEDIUM,
        impact_direction=ImpactDirection.MIXED,
        region="USA / Global",
        description="Terrorist attacks; initial spike then demand collapse fears",
        affected_producers=[],
    ),
    GeopoliticalEvent(
        event_id="2003-03-iraq-war",
        name="Iraq War (US invasion)",
        start_date=date(2003, 3, 20),
        end_date=date(2011, 12, 18),
        event_type=EventType.WAR,
        severity=Severity.HIGH,
        impact_direction=ImpactDirection.BULLISH,
        region="Middle East",
        description="US-led invasion of Iraq, sustained supply uncertainty",
        affected_producers=["Iraq"],
    ),

    # --- 2005-2008 (super-cycle bull run) ---
    GeopoliticalEvent(
        event_id="2005-08-katrina",
        name="Hurricane Katrina",
        start_date=date(2005, 8, 29),
        end_date=date(2005, 9, 30),
        event_type=EventType.NATURAL_DISASTER,
        severity=Severity.HIGH,
        impact_direction=ImpactDirection.BULLISH,
        region="USA Gulf Coast",
        description="Disrupted ~25% of US oil/gas production",
        affected_producers=["USA"],
    ),
    GeopoliticalEvent(
        event_id="2008-07-peak",
        name="Oil price peak ($147)",
        start_date=date(2008, 7, 11),
        end_date=date(2008, 7, 11),
        event_type=EventType.FINANCIAL_CRISIS,
        severity=Severity.EXTREME,
        impact_direction=ImpactDirection.BULLISH,
        region="Global",
        description="WTI hit all-time high $147.27/bbl",
        affected_producers=[],
        notes="Marker event - turning point",
    ),
    GeopoliticalEvent(
        event_id="2008-09-gfc",
        name="Global Financial Crisis",
        start_date=date(2008, 9, 15),
        end_date=date(2009, 6, 30),
        event_type=EventType.FINANCIAL_CRISIS,
        severity=Severity.EXTREME,
        impact_direction=ImpactDirection.BEARISH,
        region="Global",
        description="Lehman collapse, demand destruction, oil crashed to $32",
        affected_producers=[],
    ),

    # --- 2010-2015 ---
    GeopoliticalEvent(
        event_id="2010-12-arab-spring",
        name="Arab Spring",
        start_date=date(2010, 12, 17),
        end_date=date(2012, 12, 31),
        event_type=EventType.WAR,
        severity=Severity.HIGH,
        impact_direction=ImpactDirection.BULLISH,
        region="MENA",
        description="Regional uprisings, Libya civil war disrupted supply",
        affected_producers=["Libya", "Egypt", "Syria", "Yemen"],
    ),
    GeopoliticalEvent(
        event_id="2011-03-libya-war",
        name="Libyan Civil War (1st)",
        start_date=date(2011, 2, 15),
        end_date=date(2011, 10, 23),
        event_type=EventType.WAR,
        severity=Severity.HIGH,
        impact_direction=ImpactDirection.BULLISH,
        region="North Africa",
        description="Libya production fell from 1.6mb/d to near zero",
        affected_producers=["Libya"],
    ),
    GeopoliticalEvent(
        event_id="2012-iran-sanctions-1",
        name="Iran sanctions (Obama-era)",
        start_date=date(2012, 1, 23),
        end_date=date(2015, 7, 14),
        event_type=EventType.SANCTIONS,
        severity=Severity.MEDIUM,
        impact_direction=ImpactDirection.BULLISH,
        region="Middle East",
        description="EU oil embargo + US secondary sanctions",
        affected_producers=["Iran"],
    ),
    GeopoliticalEvent(
        event_id="2014-03-crimea",
        name="Russia annexes Crimea",
        start_date=date(2014, 2, 27),
        end_date=date(2014, 3, 18),
        event_type=EventType.WAR,
        severity=Severity.MEDIUM,
        impact_direction=ImpactDirection.MIXED,
        region="Eastern Europe",
        description="Initial spike, then sanctions on Russia (mild for oil)",
        affected_producers=["Russia"],
    ),
    GeopoliticalEvent(
        event_id="2014-11-opec-no-cut",
        name="OPEC refuses to cut (shale war)",
        start_date=date(2014, 11, 27),
        end_date=date(2016, 2, 11),
        event_type=EventType.OPEC_ACTION,
        severity=Severity.EXTREME,
        impact_direction=ImpactDirection.BEARISH,
        region="Global",
        description="Saudi Arabia maintained output to crush US shale; oil crashed to $26",
        affected_producers=["Saudi Arabia", "OPEC"],
    ),

    # --- 2016-2019 ---
    GeopoliticalEvent(
        event_id="2016-11-opec-cut",
        name="OPEC+ Production Cut Agreement",
        start_date=date(2016, 11, 30),
        end_date=date(2020, 3, 6),
        event_type=EventType.OPEC_ACTION,
        severity=Severity.HIGH,
        impact_direction=ImpactDirection.BULLISH,
        region="Global",
        description="First OPEC+Russia coordinated cuts since 2008",
        affected_producers=["OPEC", "Russia"],
    ),
    GeopoliticalEvent(
        event_id="2018-05-iran-sanctions-2",
        name="Trump re-imposes Iran sanctions",
        start_date=date(2018, 5, 8),
        end_date=date(2021, 1, 20),
        event_type=EventType.SANCTIONS,
        severity=Severity.HIGH,
        impact_direction=ImpactDirection.BULLISH,
        region="Middle East",
        description="JCPOA withdrawal, Iran exports cut from 2.5mb/d to <300kb/d",
        affected_producers=["Iran"],
    ),
    GeopoliticalEvent(
        event_id="2019-09-abqaiq",
        name="Abqaiq-Khurais attack",
        start_date=date(2019, 9, 14),
        end_date=date(2019, 9, 30),
        event_type=EventType.SUPPLY_DISRUPTION,
        severity=Severity.HIGH,
        impact_direction=ImpactDirection.BULLISH,
        region="Saudi Arabia",
        description="Drone attack knocked out 5.7mb/d (largest disruption ever); +14% in one day",
        affected_producers=["Saudi Arabia"],
    ),

    # --- 2020-2022 ---
    GeopoliticalEvent(
        event_id="2020-03-saudi-russia-war",
        name="Saudi-Russia price war",
        start_date=date(2020, 3, 8),
        end_date=date(2020, 4, 12),
        event_type=EventType.OPEC_ACTION,
        severity=Severity.EXTREME,
        impact_direction=ImpactDirection.BEARISH,
        region="Global",
        description="OPEC+ collapse, Saudi flooded market right as COVID hit",
        affected_producers=["Saudi Arabia", "Russia"],
    ),
    GeopoliticalEvent(
        event_id="2020-03-covid",
        name="COVID-19 pandemic",
        start_date=date(2020, 3, 11),
        end_date=date(2021, 12, 31),
        event_type=EventType.PANDEMIC,
        severity=Severity.EXTREME,
        impact_direction=ImpactDirection.BEARISH,
        region="Global",
        description="Demand destruction; WTI futures went negative on Apr 20, 2020",
        affected_producers=[],
        notes="WTI May 2020 contract settled at -$37.63 - unprecedented",
    ),
    GeopoliticalEvent(
        event_id="2020-04-wti-negative",
        name="WTI futures go negative",
        start_date=date(2020, 4, 20),
        end_date=date(2020, 4, 21),
        event_type=EventType.SUPPLY_DISRUPTION,
        severity=Severity.EXTREME,
        impact_direction=ImpactDirection.BEARISH,
        region="USA",
        description="Storage capacity exhaustion led to negative front-month",
        affected_producers=[],
    ),
    GeopoliticalEvent(
        event_id="2022-02-ukraine-war",
        name="Russia invades Ukraine",
        start_date=date(2022, 2, 24),
        end_date=None,  # ongoing
        event_type=EventType.WAR,
        severity=Severity.EXTREME,
        impact_direction=ImpactDirection.BULLISH,
        region="Eastern Europe",
        description="Largest war in Europe since WWII; oil briefly hit $130",
        affected_producers=["Russia"],
    ),
    GeopoliticalEvent(
        event_id="2022-06-russia-sanctions",
        name="EU bans Russian seaborne oil",
        start_date=date(2022, 6, 3),
        end_date=None,
        event_type=EventType.SANCTIONS,
        severity=Severity.HIGH,
        impact_direction=ImpactDirection.BULLISH,
        region="Europe / Russia",
        description="EU 6th sanctions package, G7 price cap followed in Dec 2022",
        affected_producers=["Russia"],
    ),
    GeopoliticalEvent(
        event_id="2022-10-opec-2mb-cut",
        name="OPEC+ surprise 2mb/d cut",
        start_date=date(2022, 10, 5),
        end_date=date(2023, 6, 30),
        event_type=EventType.OPEC_ACTION,
        severity=Severity.HIGH,
        impact_direction=ImpactDirection.BULLISH,
        region="Global",
        description="Largest cut since COVID, against US wishes",
        affected_producers=["OPEC", "Russia"],
    ),

    # --- 2023-2025 ---
    GeopoliticalEvent(
        event_id="2023-10-israel-hamas",
        name="Israel-Hamas war begins",
        start_date=date(2023, 10, 7),
        end_date=None,
        event_type=EventType.WAR,
        severity=Severity.MEDIUM,
        impact_direction=ImpactDirection.BULLISH,
        region="Middle East",
        description="Risk premium added; spillover concerns",
        affected_producers=[],
    ),
    GeopoliticalEvent(
        event_id="2023-11-houthi-attacks",
        name="Houthi Red Sea attacks begin",
        start_date=date(2023, 11, 19),
        end_date=None,
        event_type=EventType.SUPPLY_DISRUPTION,
        severity=Severity.MEDIUM,
        impact_direction=ImpactDirection.BULLISH,
        region="Red Sea / Gulf of Aden",
        description="Shipping rerouting around Cape of Good Hope, freight costs up",
        affected_producers=[],
    ),
    GeopoliticalEvent(
        event_id="2024-04-iran-israel-direct",
        name="Iran-Israel direct strikes (April)",
        start_date=date(2024, 4, 13),
        end_date=date(2024, 4, 19),
        event_type=EventType.WAR,
        severity=Severity.HIGH,
        impact_direction=ImpactDirection.BULLISH,
        region="Middle East",
        description="First direct state-on-state strikes; oil spiked then receded",
        affected_producers=["Iran"],
    ),
    GeopoliticalEvent(
        event_id="2024-10-iran-israel-2",
        name="Iran-Israel direct strikes (October)",
        start_date=date(2024, 10, 1),
        end_date=date(2024, 10, 26),
        event_type=EventType.WAR,
        severity=Severity.HIGH,
        impact_direction=ImpactDirection.BULLISH,
        region="Middle East",
        description="Second round of direct strikes",
        affected_producers=["Iran"],
    ),
    GeopoliticalEvent(
        event_id="2024-09-pager-attack",
        name="Hezbollah pager/walkie-talkie attacks",
        start_date=date(2024, 9, 17),
        end_date=date(2024, 9, 18),
        event_type=EventType.WAR,
        severity=Severity.MEDIUM,
        impact_direction=ImpactDirection.BULLISH,
        region="Lebanon / Middle East",
        description="Israeli operation targeting Hezbollah comms; regional escalation fears",
        affected_producers=[],
    ),
    GeopoliticalEvent(
        event_id="2024-11-trump-elected",
        name="Trump elected US President (2nd term)",
        start_date=date(2024, 11, 5),
        end_date=date(2024, 11, 6),
        event_type=EventType.DIPLOMATIC,
        severity=Severity.MEDIUM,
        impact_direction=ImpactDirection.MIXED,
        region="USA",
        description="Trump victory shifted expectations: tougher on Iran, OPEC pressure, sanctions",
        affected_producers=["Iran", "Russia", "Venezuela"],
        notes="Initial reaction muted but framed all 2025-26 policy",
    ),
    GeopoliticalEvent(
        event_id="2025-01-trump-inauguration",
        name="Trump inauguration & maximum-pressure resumption",
        start_date=date(2025, 1, 20),
        end_date=date(2025, 2, 28),
        event_type=EventType.SANCTIONS,
        severity=Severity.MEDIUM,
        impact_direction=ImpactDirection.BULLISH,
        region="Global",
        description="Day-one EOs reviving maximum-pressure on Iran; secondary sanctions on Chinese refiners",
        affected_producers=["Iran"],
    ),
    GeopoliticalEvent(
        event_id="2025-04-trump-iran-deadline",
        name="Trump Iran deal ultimatum (60-day)",
        start_date=date(2025, 4, 1),
        end_date=date(2025, 6, 12),
        event_type=EventType.DIPLOMATIC,
        severity=Severity.MEDIUM,
        impact_direction=ImpactDirection.BULLISH,
        region="USA / Iran",
        description="Public ultimatum from Trump; failed Geneva nuclear negotiations led to June 2025 war",
        affected_producers=["Iran"],
    ),
    GeopoliticalEvent(
        event_id="2025-06-israel-iran-12day",
        name="June 2025 Israel-Iran 12-day air war",
        start_date=date(2025, 6, 13),
        end_date=date(2025, 6, 24),
        event_type=EventType.WAR,
        severity=Severity.EXTREME,
        impact_direction=ImpactDirection.BULLISH,
        region="Middle East",
        description="Direct Israel-Iran air war; Israeli strikes hit Bandar Abbas near Hormuz; "
                    "Brent surged but receded after de-escalation. Brent peaked ~$80 from ~$68 pre-war.",
        affected_producers=["Iran"],
        notes="12-day conflict ended with informal ceasefire; precursor to 2026 war",
    ),
    GeopoliticalEvent(
        event_id="2025-09-snapback-sanctions",
        name="UN snapback sanctions on Iran",
        start_date=date(2025, 9, 27),
        end_date=None,
        event_type=EventType.SANCTIONS,
        severity=Severity.HIGH,
        impact_direction=ImpactDirection.BULLISH,
        region="Global",
        description="France/Germany/UK triggered JCPOA snapback citing significant non-performance",
        affected_producers=["Iran"],
    ),
    GeopoliticalEvent(
        event_id="2025-10-china-iran-oil-infra",
        name="Sinosure-Iran $8.4B oil-for-infrastructure deal exposed",
        start_date=date(2025, 10, 15),
        end_date=date(2025, 10, 15),
        event_type=EventType.DIPLOMATIC,
        severity=Severity.MEDIUM,
        impact_direction=ImpactDirection.MIXED,
        region="China / Iran",
        description="WSJ revealed Chinese state-backed financing of Iran through Sinosure; "
                    "highlighted resilience of Iran's oil exports to China via teapot refiners",
        affected_producers=["Iran", "China"],
    ),
    GeopoliticalEvent(
        event_id="2025-12-iran-prepares-hormuz",
        name="Iran reportedly prepares Strait of Hormuz mining",
        start_date=date(2025, 12, 1),
        end_date=date(2026, 2, 27),
        event_type=EventType.SUPPLY_DISRUPTION,
        severity=Severity.HIGH,
        impact_direction=ImpactDirection.BULLISH,
        region="Persian Gulf",
        description="Reuters: US intel showed Iran preparing naval mines for Hormuz; insurance premiums up",
        affected_producers=["All Gulf producers"],
    ),
    GeopoliticalEvent(
        event_id="2026-02-iran-export-surge",
        name="Iran pre-war oil export surge (3x normal)",
        start_date=date(2026, 2, 15),
        end_date=date(2026, 2, 20),
        event_type=EventType.SUPPLY_DISRUPTION,
        severity=Severity.MEDIUM,
        impact_direction=ImpactDirection.BEARISH,
        region="Iran",
        description="Iran tripled exports and drew down storage as war fears mounted; "
                    "Saudi Arabia did similar moves",
        affected_producers=["Iran", "Saudi Arabia"],
    ),
    GeopoliticalEvent(
        event_id="2026-02-epic-fury",
        name="Operation Epic Fury (US-Israel strikes on Iran)",
        start_date=date(2026, 2, 28),
        end_date=None,
        event_type=EventType.WAR,
        severity=Severity.EXTREME,
        impact_direction=ImpactDirection.BULLISH,
        region="Middle East",
        description="Coordinated US-Israel air war: command centers, IRGC HQ, nuclear sites, navy. "
                    "Supreme Leader Khamenei killed. Iran retaliated against US bases in UAE, Qatar, Bahrain.",
        affected_producers=["Iran"],
        notes="Brent jumped from ~$72 to ~$80-82 in 2 days, peaked near $120 in March (+55%)",
    ),
    GeopoliticalEvent(
        event_id="2026-03-hormuz-closure",
        name="Iran closes Strait of Hormuz (2026)",
        start_date=date(2026, 3, 4),
        end_date=None,
        event_type=EventType.SUPPLY_DISRUPTION,
        severity=Severity.EXTREME,
        impact_direction=ImpactDirection.BULLISH,
        region="Strait of Hormuz",
        description="Iran declared Hormuz closed; threatened ships; UKMTO reported >12 ship attacks. "
                    "IEA: 'largest supply disruption in history of oil market.' "
                    "~25% of seaborne oil, ~20% of LNG normally pass through.",
        affected_producers=["Saudi Arabia", "UAE", "Kuwait", "Qatar", "Iraq", "Iran"],
        notes="Brent peaked near $120 (+55% from pre-war); insurance up 4-6x",
    ),
    GeopoliticalEvent(
        event_id="2026-03-iran-missile-attacks-gulf",
        name="Iranian missile/drone attacks on US bases in Gulf states",
        start_date=date(2026, 3, 1),
        end_date=date(2026, 3, 20),
        event_type=EventType.WAR,
        severity=Severity.HIGH,
        impact_direction=ImpactDirection.BULLISH,
        region="UAE / Qatar / Bahrain",
        description="Iran retaliated with missiles/drones on US bases and infrastructure across "
                    "Gulf Cooperation Council states; Qatar Ras Laffan LNG affected (force majeure)",
        affected_producers=["Qatar", "UAE", "Bahrain"],
    ),
    GeopoliticalEvent(
        event_id="2026-03-trump-strait-of-trump",
        name="Trump proposes renaming Hormuz to 'Strait of Trump'",
        start_date=date(2026, 3, 27),
        end_date=date(2026, 3, 27),
        event_type=EventType.DIPLOMATIC,
        severity=Severity.LOW,
        impact_direction=ImpactDirection.MIXED,
        region="USA / Persian Gulf",
        description="Trump expressed desire to seize control of the strait and rename it; "
                    "headline-driven volatility but limited real impact",
        affected_producers=[],
        notes="Example of Trump statements driving short-term oil volatility in 2026",
    ),
    GeopoliticalEvent(
        event_id="2026-03-trump-postpones-attacks",
        name="Trump postpones attacks (3/23) — suspicious oil futures bets",
        start_date=date(2026, 3, 23),
        end_date=date(2026, 3, 23),
        event_type=EventType.DIPLOMATIC,
        severity=Severity.MEDIUM,
        impact_direction=ImpactDirection.BEARISH,
        region="USA / Iran",
        description="Trump announced 2-week postponement of attacks for talks. $580M in short oil bets "
                    "placed 15 min prior raised insider-trading concerns. Oil sold off sharply on news.",
        affected_producers=[],
        notes="One of multiple Trump-statement-driven moves of >5% intraday during 2026 war",
    ),
    GeopoliticalEvent(
        event_id="2026-04-iran-us-ceasefire",
        name="Iran-US ceasefire announced",
        start_date=date(2026, 4, 8),
        end_date=date(2026, 4, 19),
        event_type=EventType.DIPLOMATIC,
        severity=Severity.HIGH,
        impact_direction=ImpactDirection.BEARISH,
        region="Global",
        description="Two-week ceasefire with partial Hormuz reopening; oil fell >10% on April 17 "
                    "when Iran foreign minister declared strait 'fully open'",
        affected_producers=["Iran"],
    ),
    GeopoliticalEvent(
        event_id="2026-04-us-iran-blockade",
        name="US Navy blockade of Iranian ports",
        start_date=date(2026, 4, 13),
        end_date=None,
        event_type=EventType.SUPPLY_DISRUPTION,
        severity=Severity.HIGH,
        impact_direction=ImpactDirection.BULLISH,
        region="Iran",
        description="US Navy enforced blockade of Iranian export terminals; Iran reimposed Hormuz "
                    "restrictions in response. Net effect: kept Brent in $80-90 range vs pre-crisis $72.",
        affected_producers=["Iran"],
    ),
    GeopoliticalEvent(
        event_id="2026-04-iran-yuan-tariffs",
        name="Iran proposes yuan-denominated Hormuz tariffs",
        start_date=date(2026, 4, 22),
        end_date=None,
        event_type=EventType.DIPLOMATIC,
        severity=Severity.MEDIUM,
        impact_direction=ImpactDirection.MIXED,
        region="Persian Gulf / China",
        description="Iran's parliament moved to formalize transit tariffs (~$40-50B/yr) "
                    "paid in yuan, accelerating petrodollar-to-petroyuan shift. "
                    "Test of bifurcated oil market: compliant vessels in yuan vs dollar-only",
        affected_producers=["Iran", "Saudi Arabia", "UAE", "Qatar", "Kuwait"],
        notes="Structural shift: PetroYuan trade architecture (CIPS) handling $134B/day by March 2026",
    ),
    GeopoliticalEvent(
        event_id="2026-05-project-freedom",
        name="Operation Project Freedom (US Navy escort missions)",
        start_date=date(2026, 5, 4),
        end_date=None,
        event_type=EventType.WAR,
        severity=Severity.MEDIUM,
        impact_direction=ImpactDirection.BEARISH,
        region="Persian Gulf",
        description="US Navy began escorting commercial vessels out of Gulf; Iran threatened "
                    "this constituted ceasefire violation. Trump paused on May 6 citing progress.",
        affected_producers=[],
    ),
]


def get_events_in_range(start: date, end: date) -> list[GeopoliticalEvent]:
    """Return events that overlap a date range"""
    out = []
    for e in EVENTS:
        e_end = e.end_date or date.today()
        if e.start_date <= end and e_end >= start:
            out.append(e)
    return out


def is_event_active(check_date: date, event_types: Optional[list[EventType]] = None) -> bool:
    """Check if any event is active on a date"""
    for e in EVENTS:
        if event_types and e.event_type not in event_types:
            continue
        e_end = e.end_date or date.today()
        if e.start_date <= check_date <= e_end:
            return True
    return False


def label_dates(dates: list[date]) -> list[dict]:
    """Label each date with active event metadata. Used for regime training."""
    labels = []
    for d in dates:
        active = [e for e in EVENTS
                  if e.start_date <= d <= (e.end_date or date.today())]
        labels.append({
            "date": d.isoformat(),
            "n_active_events": len(active),
            "has_war": any(e.event_type == EventType.WAR for e in active),
            "has_sanctions": any(e.event_type == EventType.SANCTIONS for e in active),
            "has_supply_disruption": any(e.event_type == EventType.SUPPLY_DISRUPTION for e in active),
            "max_severity": max((e.severity.value for e in active), default="none"),
            "active_event_ids": [e.event_id for e in active],
        })
    return labels


def export_to_json(path: str) -> None:
    """Export full event database to JSON"""
    data = [e.to_dict() for e in EVENTS]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    print(f"Total events in database: {len(EVENTS)}")
    print(f"Date range: {min(e.start_date for e in EVENTS)} to "
          f"{max((e.end_date or date.today()) for e in EVENTS)}")
    print("\nBy type:")
    from collections import Counter
    type_counts = Counter(e.event_type.value for e in EVENTS)
    for t, c in type_counts.most_common():
        print(f"  {t}: {c}")
