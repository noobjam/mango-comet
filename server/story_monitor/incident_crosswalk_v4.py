"""Deterministic overlap crosswalk from an old incident release to V4 replay."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import pandas as pd


SCHEMA_VERSION = "incident-overlap-crosswalk-v4/1"
MEMBERSHIP_FILENAME = "incident_membership.parquet"
CROSSWALK_COLUMNS = (
    "old_incident_id",
    "new_incident_id",
    "match_status",
    "old_field_week_count",
    "new_field_week_count",
    "overlap_field_week_count",
    "union_field_week_count",
    "jaccard_similarity",
    "old_coverage_fraction",
    "new_coverage_fraction",
    "schema_version",
)
_MEMBERSHIP_COLUMNS = ("incident_id", "timeline_bucket", "field_id")
_COUNT_COLUMNS = (
    "old_field_week_count",
    "new_field_week_count",
    "overlap_field_week_count",
    "union_field_week_count",
)
_FRACTION_COLUMNS = (
    "jaccard_similarity",
    "old_coverage_fraction",
    "new_coverage_fraction",
)

MembershipSource: TypeAlias = pd.DataFrame | Path | str


def build_incident_crosswalk_v4(
    old_memberships: MembershipSource,
    new_memberships: MembershipSource,
) -> pd.DataFrame:
    """Compare incident field/week sets without changing either identity space.

    Each positive-overlap old/new pair receives one row. An incident with no
    positive overlap receives one ``unmatched_old`` or ``unmatched_new`` row.
    Coverage is directional: overlap divided by the corresponding incident's
    canonical distinct ``(timeline_bucket, field_id)`` count.
    """
    old = _prepare_memberships(old_memberships, "old").rename(
        columns={"incident_id": "old_incident_id"}
    )
    new = _prepare_memberships(new_memberships, "new").rename(
        columns={"incident_id": "new_incident_id"}
    )
    old_counts = (
        old.groupby("old_incident_id", sort=True)
        .size()
        .rename("old_field_week_count")
        .reset_index()
    )
    new_counts = (
        new.groupby("new_incident_id", sort=True)
        .size()
        .rename("new_field_week_count")
        .reset_index()
    )

    joined = old.merge(
        new,
        on=["timeline_bucket", "field_id"],
        how="inner",
        validate="many_to_many",
    )
    overlaps = (
        joined.groupby(["old_incident_id", "new_incident_id"], sort=True)
        .size()
        .rename("overlap_field_week_count")
        .reset_index()
    )
    overlaps = overlaps.merge(
        old_counts, on="old_incident_id", how="left", validate="many_to_one"
    ).merge(
        new_counts, on="new_incident_id", how="left", validate="many_to_one"
    )

    frames: list[pd.DataFrame] = []
    if not overlaps.empty:
        overlaps["union_field_week_count"] = (
            overlaps["old_field_week_count"]
            + overlaps["new_field_week_count"]
            - overlaps["overlap_field_week_count"]
        )
        overlaps["jaccard_similarity"] = (
            overlaps["overlap_field_week_count"]
            / overlaps["union_field_week_count"]
        )
        overlaps["old_coverage_fraction"] = (
            overlaps["overlap_field_week_count"]
            / overlaps["old_field_week_count"]
        )
        overlaps["new_coverage_fraction"] = (
            overlaps["overlap_field_week_count"]
            / overlaps["new_field_week_count"]
        )
        overlaps["match_status"] = "overlap"
        frames.append(overlaps)

    matched_old = set(overlaps["old_incident_id"].astype(str))
    unmatched_old = old_counts[
        ~old_counts["old_incident_id"].isin(matched_old)
    ].copy()
    if not unmatched_old.empty:
        unmatched_old["new_incident_id"] = pd.NA
        unmatched_old["new_field_week_count"] = 0
        unmatched_old["overlap_field_week_count"] = 0
        unmatched_old["union_field_week_count"] = unmatched_old[
            "old_field_week_count"
        ]
        for column in _FRACTION_COLUMNS:
            unmatched_old[column] = 0.0
        unmatched_old["match_status"] = "unmatched_old"
        frames.append(unmatched_old)

    matched_new = set(overlaps["new_incident_id"].astype(str))
    unmatched_new = new_counts[
        ~new_counts["new_incident_id"].isin(matched_new)
    ].copy()
    if not unmatched_new.empty:
        unmatched_new["old_incident_id"] = pd.NA
        unmatched_new["old_field_week_count"] = 0
        unmatched_new["overlap_field_week_count"] = 0
        unmatched_new["union_field_week_count"] = unmatched_new[
            "new_field_week_count"
        ]
        for column in _FRACTION_COLUMNS:
            unmatched_new[column] = 0.0
        unmatched_new["match_status"] = "unmatched_new"
        frames.append(unmatched_new)

    if not frames:
        return _empty_crosswalk()
    output = pd.concat(frames, ignore_index=True, sort=False)
    output["schema_version"] = SCHEMA_VERSION
    for column in ("old_incident_id", "new_incident_id", "match_status"):
        output[column] = output[column].astype("string")
    for column in _COUNT_COLUMNS:
        output[column] = output[column].astype("int64")
    for column in _FRACTION_COLUMNS:
        output[column] = output[column].astype("float64")
    output["_sort_status"] = output["match_status"].map(
        {"overlap": 0, "unmatched_old": 1, "unmatched_new": 2}
    )
    return (
        output.sort_values(
            ["_sort_status", "old_incident_id", "new_incident_id"],
            kind="mergesort",
            na_position="last",
        )
        .drop(columns="_sort_status")
        .loc[:, CROSSWALK_COLUMNS]
        .reset_index(drop=True)
    )


def _prepare_memberships(source: MembershipSource, side: str) -> pd.DataFrame:
    frame = _read_memberships(source, side)
    missing = sorted(set(_MEMBERSHIP_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(
            f"{side} incident membership is missing columns: {', '.join(missing)}"
        )
    output = frame.loc[:, _MEMBERSHIP_COLUMNS].copy()
    for column in ("incident_id", "field_id"):
        if output[column].isna().any():
            raise ValueError(f"{side} incident membership contains null {column}")
        output[column] = output[column].astype(str).str.strip()
        if output[column].eq("").any():
            raise ValueError(f"{side} incident membership contains blank {column}")
    try:
        output["timeline_bucket"] = output["timeline_bucket"].map(
            _normalize_timeline_bucket
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{side} incident membership contains an invalid timeline_bucket"
        ) from exc
    if output["timeline_bucket"].isna().any():
        raise ValueError(
            f"{side} incident membership contains a null timeline_bucket"
        )
    duplicate = output.duplicated(list(_MEMBERSHIP_COLUMNS), keep=False)
    if duplicate.any() and side == "new":
        examples = output.loc[duplicate, _MEMBERSHIP_COLUMNS].head(3).to_dict(
            orient="records"
        )
        raise ValueError(
            f"{side} incident membership is not canonical by story, week, and "
            f"field: {examples}"
        )
    if duplicate.any():
        # The audit-only V3 release may predate the canonical one-row-per-field
        # fix.  Its duplicate roles must not block replay or inflate overlap,
        # and they never influence the new identity space.
        output = output.drop_duplicates(list(_MEMBERSHIP_COLUMNS), keep="first")
    return output.sort_values(
        list(_MEMBERSHIP_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)


def _read_memberships(source: MembershipSource, side: str) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        path = path / MEMBERSHIP_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{side} incident membership does not exist: {path}"
        )
    return pd.read_parquet(path)


def _normalize_timeline_bucket(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _empty_crosswalk() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "old_incident_id": pd.Series(dtype="string"),
            "new_incident_id": pd.Series(dtype="string"),
            "match_status": pd.Series(dtype="string"),
            **{column: pd.Series(dtype="int64") for column in _COUNT_COLUMNS},
            **{column: pd.Series(dtype="float64") for column in _FRACTION_COLUMNS},
            "schema_version": pd.Series(dtype="object"),
        },
        columns=CROSSWALK_COLUMNS,
    )


__all__ = [
    "CROSSWALK_COLUMNS",
    "MEMBERSHIP_FILENAME",
    "SCHEMA_VERSION",
    "build_incident_crosswalk_v4",
]
