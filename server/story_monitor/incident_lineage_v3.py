"""Map exposure split/merge lineage onto crop-specific incident identities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd


LINEAGE_SCHEMA_VERSION = "crop-incident-lineage-v3/1"
LINEAGE_COLUMNS = (
    "lineage_id",
    "timeline_bucket",
    "lineage_type",
    "crop_name_normalized",
    "parent_exposure_id",
    "child_exposure_id",
    "parent_incident_id",
    "child_incident_id",
    "previous_component_id",
    "current_component_id",
    "score",
    "schema_version",
)
METADATA_COLUMNS = (
    "incident_id",
    "exposure_id",
    "crop_name_normalized",
    "split_count",
    "split_out_count",
    "split_from_count",
    "merge_count",
    "merge_in_count",
    "merge_out_count",
    "merged_into_incident_id",
    "merged_week",
    "schema_version",
)
_EXPOSURE_REQUIRED = (
    "timeline_bucket",
    "parent_exposure_id",
    "child_exposure_id",
    "lineage_type",
    "score",
    "previous_component_id",
    "current_component_id",
)
_CATALOG_REQUIRED = ("exposure_id", "crop_name_normalized", "incident_id")
_SUPPORTED_TYPES = frozenset({"split", "merge"})
_IGNORED_TYPES = frozenset({"related_unmatched"})


@dataclass(frozen=True)
class IncidentLineageArtifacts:
    """Crop-specific edges plus one lifecycle metadata row per incident."""

    lineage: pd.DataFrame
    incident_metadata: pd.DataFrame


def build_incident_lineage_v3(
    exposure_lineage: pd.DataFrame,
    incident_catalog: pd.DataFrame,
    component_memberships: pd.DataFrame,
) -> IncidentLineageArtifacts:
    """Map split/merge edges only where parent and child share a crop.

    ``split_count`` and ``merge_count`` count distinct lineage edges touching an
    incident.  Directional columns distinguish a parent event from a child
    event.  ``merged_into_*`` is populated only on the outgoing merge parent,
    making it safe to feed into a terminal lifecycle transition.
    """
    catalog = _prepare_catalog(incident_catalog)
    source = _prepare_exposure_lineage(exposure_lineage)
    component_crops = _prepare_component_crops(component_memberships)
    supported = source[source["lineage_type"].isin(_SUPPORTED_TYPES)].copy()
    if supported.empty or catalog.empty or component_crops.empty:
        return IncidentLineageArtifacts(
            pd.DataFrame(columns=LINEAGE_COLUMNS),
            _metadata(catalog, pd.DataFrame(columns=LINEAGE_COLUMNS)),
        )

    parent_catalog = catalog.rename(
        columns={
            "exposure_id": "parent_exposure_id",
            "incident_id": "parent_incident_id",
        }
    )
    child_catalog = catalog.rename(
        columns={
            "exposure_id": "child_exposure_id",
            "incident_id": "child_incident_id",
        }
    )
    previous_crops = component_crops.rename(
        columns={"component_id": "previous_component_id"}
    )
    current_crops = component_crops.rename(
        columns={"component_id": "current_component_id"}
    )
    edge_crops = supported.merge(
        previous_crops,
        on="previous_component_id",
        how="inner",
        validate="many_to_many",
    ).merge(
        current_crops,
        on=["current_component_id", "crop_name_normalized"],
        how="inner",
        validate="many_to_one",
    )
    mapped = edge_crops.merge(
        parent_catalog,
        on=["parent_exposure_id", "crop_name_normalized"],
        how="inner",
        validate="many_to_one",
    ).merge(
        child_catalog,
        on=["child_exposure_id", "crop_name_normalized"],
        how="inner",
        validate="many_to_one",
    )
    if mapped.empty:
        return IncidentLineageArtifacts(
            pd.DataFrame(columns=LINEAGE_COLUMNS),
            _metadata(catalog, pd.DataFrame(columns=LINEAGE_COLUMNS)),
        )

    mapped["lineage_id"] = mapped.apply(_lineage_id, axis=1)
    mapped["schema_version"] = LINEAGE_SCHEMA_VERSION
    lineage = mapped.loc[:, LINEAGE_COLUMNS].sort_values(
        [
            "timeline_bucket",
            "lineage_type",
            "crop_name_normalized",
            "parent_incident_id",
            "child_incident_id",
            "previous_component_id",
            "current_component_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    validate_incident_lineage(lineage)
    return IncidentLineageArtifacts(lineage, _metadata(catalog, lineage))


def _prepare_component_crops(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        frame,
        ("component_id", "crop_name", "membership_role"),
        "component memberships for incident lineage",
    )
    if frame.empty:
        return pd.DataFrame(columns=["component_id", "crop_name_normalized"])
    output = frame[
        frame["membership_role"].astype(str).isin(
            {"pressure_core", "impact_lag"}
        )
    ].loc[:, ["component_id", "crop_name"]].copy()
    output["component_id"] = output["component_id"].astype(str)
    output["crop_name_normalized"] = output["crop_name"].map(_crop)
    return output.loc[:, ["component_id", "crop_name_normalized"]].drop_duplicates()


def remap_incident_lineage_segments(
    lineage: pd.DataFrame,
    incident_weekly_state: pd.DataFrame,
    segmented_catalog: pd.DataFrame,
) -> IncidentLineageArtifacts:
    """Attach each exposure edge to the crop-story segment active that week."""
    _require_columns(lineage, LINEAGE_COLUMNS, "incident lineage")
    _require_columns(
        incident_weekly_state,
        ("timeline_bucket", "incident_id", "base_incident_id"),
        "segmented incident weekly state",
    )
    catalog = _prepare_segment_catalog(segmented_catalog)
    if incident_weekly_state.empty or lineage.empty:
        empty = pd.DataFrame(columns=LINEAGE_COLUMNS)
        return IncidentLineageArtifacts(empty, _metadata(catalog, empty))
    weekly = incident_weekly_state.loc[
        :, ["timeline_bucket", "incident_id", "base_incident_id"]
    ].copy()
    weekly["timeline_bucket"] = pd.to_datetime(
        weekly["timeline_bucket"], errors="raise"
    ).dt.normalize()
    if weekly.duplicated(["timeline_bucket", "base_incident_id"]).any():
        raise ValueError(
            "segmented incident state maps a base incident to multiple active segments"
        )
    active = {
        (str(row["base_incident_id"]), pd.Timestamp(row["timeline_bucket"]).normalize()):
            str(row["incident_id"])
        for row in weekly.to_dict("records")
    }
    records: list[dict[str, Any]] = []
    for source in lineage.to_dict("records"):
        week = pd.Timestamp(source["timeline_bucket"]).normalize()
        parent = active.get((str(source["parent_incident_id"]), week))
        child = active.get((str(source["child_incident_id"]), week))
        if parent is None or child is None:
            continue
        record = {
            **source,
            "timeline_bucket": week,
            "parent_incident_id": parent,
            "child_incident_id": child,
            "schema_version": LINEAGE_SCHEMA_VERSION,
        }
        record["lineage_id"] = _lineage_id(pd.Series(record))
        records.append(record)
    remapped = pd.DataFrame(records, columns=LINEAGE_COLUMNS)
    if not remapped.empty:
        remapped = remapped.sort_values(
            [
                "timeline_bucket", "lineage_type", "crop_name_normalized",
                "parent_incident_id", "child_incident_id",
                "previous_component_id", "current_component_id",
            ],
            kind="mergesort",
        ).reset_index(drop=True)
    validate_incident_lineage(remapped)
    return IncidentLineageArtifacts(remapped, _metadata(catalog, remapped))


def validate_incident_lineage(lineage: pd.DataFrame) -> None:
    """Reject duplicate, self-referential, or cyclic incident lineage."""
    _require_columns(lineage, LINEAGE_COLUMNS, "incident lineage")
    if lineage.empty:
        return
    _require_nonblank(
        lineage,
        (
            "lineage_id",
            "lineage_type",
            "crop_name_normalized",
            "parent_incident_id",
            "child_incident_id",
        ),
        "incident lineage",
    )
    natural_key = (
        "timeline_bucket",
        "lineage_type",
        "crop_name_normalized",
        "parent_incident_id",
        "child_incident_id",
    )
    if lineage.duplicated(list(natural_key)).any() or lineage["lineage_id"].duplicated().any():
        raise ValueError("incident lineage contains duplicate crop-specific edges")
    if (~lineage["lineage_type"].isin(_SUPPORTED_TYPES)).any():
        raise ValueError("incident lineage contains an unsupported lineage_type")
    if (
        lineage["parent_incident_id"].astype(str)
        == lineage["child_incident_id"].astype(str)
    ).any():
        raise ValueError("incident lineage contains a self-cycle")
    _assert_acyclic(
        zip(
            lineage["parent_incident_id"].astype(str),
            lineage["child_incident_id"].astype(str),
        )
    )


def _prepare_exposure_lineage(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    _apply_alias(output, "parent_exposure_id", "source_exposure_id")
    _apply_alias(output, "child_exposure_id", "target_exposure_id")
    _require_columns(output, _EXPOSURE_REQUIRED, "exposure lineage")
    if output.empty:
        return output
    _require_nonblank(
        output,
        (
            "parent_exposure_id",
            "child_exposure_id",
            "lineage_type",
            "previous_component_id",
            "current_component_id",
        ),
        "exposure lineage",
    )
    output["timeline_bucket"] = pd.to_datetime(
        output["timeline_bucket"], errors="coerce"
    ).dt.normalize()
    if output["timeline_bucket"].isna().any():
        raise ValueError("exposure lineage contains invalid timeline_bucket values")
    output["lineage_type"] = output["lineage_type"].astype(str).str.strip().str.lower()
    unknown = sorted(set(output["lineage_type"]) - _SUPPORTED_TYPES - _IGNORED_TYPES)
    if unknown:
        raise ValueError("exposure lineage contains unsupported types: " + ", ".join(unknown))
    output["score"] = pd.to_numeric(output["score"], errors="coerce")
    if not np.isfinite(output["score"].to_numpy(dtype=float)).all():
        raise ValueError("exposure lineage scores must be finite")
    key = (
        "timeline_bucket",
        "parent_exposure_id",
        "child_exposure_id",
        "lineage_type",
        "previous_component_id",
        "current_component_id",
    )
    if output.duplicated(list(key)).any():
        raise ValueError("exposure lineage contains duplicate component edges")
    return output


def _prepare_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, _CATALOG_REQUIRED, "incident catalog")
    output = frame.loc[:, _CATALOG_REQUIRED].copy()
    if output.empty:
        return output
    _require_nonblank(output, _CATALOG_REQUIRED, "incident catalog")
    output["exposure_id"] = output["exposure_id"].astype(str)
    output["incident_id"] = output["incident_id"].astype(str)
    output["crop_name_normalized"] = output["crop_name_normalized"].map(_crop)
    if output.duplicated(["exposure_id", "crop_name_normalized"]).any():
        raise ValueError("incident catalog duplicates exposure and crop")
    if output["incident_id"].duplicated().any():
        raise ValueError("incident catalog maps one incident_id more than once")
    return output.sort_values(
        ["exposure_id", "crop_name_normalized", "incident_id"], kind="mergesort"
    ).reset_index(drop=True)


def _prepare_segment_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, _CATALOG_REQUIRED, "segmented incident catalog")
    output = frame.loc[:, _CATALOG_REQUIRED].copy()
    if output.empty:
        return output
    _require_nonblank(output, _CATALOG_REQUIRED, "segmented incident catalog")
    output["exposure_id"] = output["exposure_id"].astype(str)
    output["incident_id"] = output["incident_id"].astype(str)
    output["crop_name_normalized"] = output["crop_name_normalized"].map(_crop)
    if output["incident_id"].duplicated().any():
        raise ValueError("segmented incident catalog maps one incident_id more than once")
    return output.sort_values(
        ["exposure_id", "crop_name_normalized", "incident_id"], kind="mergesort"
    ).reset_index(drop=True)


def _metadata(catalog: pd.DataFrame, lineage: pd.DataFrame) -> pd.DataFrame:
    metadata = catalog.copy()
    for name in (
        "split_count",
        "split_out_count",
        "split_from_count",
        "merge_count",
        "merge_in_count",
        "merge_out_count",
    ):
        metadata[name] = 0
    metadata["merged_into_incident_id"] = None
    metadata["merged_week"] = pd.NaT
    metadata["schema_version"] = LINEAGE_SCHEMA_VERSION
    if lineage.empty:
        return metadata.loc[:, METADATA_COLUMNS].sort_values(
            ["incident_id"], kind="mergesort"
        ).reset_index(drop=True)

    counts: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    outgoing_merges: dict[str, list[tuple[str, pd.Timestamp]]] = defaultdict(list)
    for row in lineage.to_dict("records"):
        edge_id = str(row["lineage_id"])
        parent = str(row["parent_incident_id"])
        child = str(row["child_incident_id"])
        kind = str(row["lineage_type"])
        counts[parent][f"{kind}_all"].add(edge_id)
        counts[child][f"{kind}_all"].add(edge_id)
        counts[parent][f"{kind}_out"].add(edge_id)
        counts[child][f"{kind}_in"].add(edge_id)
        if kind == "merge":
            outgoing_merges[parent].append(
                (child, pd.Timestamp(row["timeline_bucket"]).normalize())
            )
    for index, row in metadata.iterrows():
        incident_id = str(row["incident_id"])
        metadata.at[index, "split_count"] = len(counts[incident_id]["split_all"])
        metadata.at[index, "split_out_count"] = len(counts[incident_id]["split_out"])
        metadata.at[index, "split_from_count"] = len(counts[incident_id]["split_in"])
        metadata.at[index, "merge_count"] = len(counts[incident_id]["merge_all"])
        metadata.at[index, "merge_in_count"] = len(counts[incident_id]["merge_in"])
        metadata.at[index, "merge_out_count"] = len(counts[incident_id]["merge_out"])
        targets = sorted(set(outgoing_merges.get(incident_id, ())))
        if len(targets) > 1:
            raise ValueError(
                f"incident {incident_id} merges into multiple targets or weeks"
            )
        if targets:
            metadata.at[index, "merged_into_incident_id"] = targets[0][0]
            metadata.at[index, "merged_week"] = targets[0][1]
    integer_columns = [
        "split_count",
        "split_out_count",
        "split_from_count",
        "merge_count",
        "merge_in_count",
        "merge_out_count",
    ]
    metadata[integer_columns] = metadata[integer_columns].astype("int64")
    return metadata.loc[:, METADATA_COLUMNS].sort_values(
        ["incident_id"], kind="mergesort"
    ).reset_index(drop=True)


def _assert_acyclic(edges: Iterable[tuple[str, str]]) -> None:
    graph: dict[str, set[str]] = defaultdict(set)
    for parent, child in edges:
        graph[parent].add(child)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("incident lineage contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for child in sorted(graph.get(node, ())):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in sorted(graph):
        visit(node)


def _lineage_id(row: pd.Series) -> str:
    values = (
        pd.Timestamp(row["timeline_bucket"]).date().isoformat(),
        row["lineage_type"],
        row["crop_name_normalized"],
        row["parent_incident_id"],
        row["child_incident_id"],
        row["previous_component_id"],
        row["current_component_id"],
    )
    raw = "\x1f".join(str(value) for value in values)
    return "incident_lineage_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _apply_alias(frame: pd.DataFrame, canonical: str, alias: str) -> None:
    if canonical not in frame and alias in frame:
        frame[canonical] = frame[alias]
    elif canonical in frame and alias in frame:
        left = frame[canonical].fillna("").astype(str)
        right = frame[alias].fillna("").astype(str)
        if not left.equals(right):
            raise ValueError(f"exposure lineage has conflicting {canonical}/{alias}")


def _crop(value: Any) -> str:
    return (
        re.sub(r"[^a-z0-9]+", "_", str(value or "unknown_crop").strip().lower())
        .strip("_")
        or "unknown_crop"
    )


def _require_columns(frame: pd.DataFrame, names: Iterable[str], label: str) -> None:
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _require_nonblank(
    frame: pd.DataFrame, names: Iterable[str], label: str
) -> None:
    for name in names:
        values = frame[name]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"{label}.{name} contains null or blank values")


build_crop_incident_lineage = build_incident_lineage_v3


__all__ = [
    "IncidentLineageArtifacts",
    "LINEAGE_COLUMNS",
    "LINEAGE_SCHEMA_VERSION",
    "METADATA_COLUMNS",
    "build_crop_incident_lineage",
    "build_incident_lineage_v3",
    "remap_incident_lineage_segments",
    "validate_incident_lineage",
]
