from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from types import EllipsisType
from typing import Any

import dagster as dg


@dataclass(frozen=True, kw_only=True, eq=False)
class Layer[R]:
    """A group of assets with shared key space, output type and depenency sources."""

    name: str
    description: str | None = None
    key_prefix: tuple[str, ...]
    output_names: frozenset[str]
    python_type: type[R]
    partitions_def: dg.PartitionsDefinition | None = None
    sources: "tuple[LayerDependency, ...]" = field(default_factory=tuple)

    def __contains__(self, output_name: str) -> bool:
        return output_name in self.output_names

    def resolve_key(self, output_name: str) -> dg.AssetKey:
        if output_name not in self:
            raise ValueError(
                f"{output_name!r} is not a known output of layer {self.name!r}."
            )
        return dg.AssetKey(output_name).with_prefix(self.key_prefix)

    def resolve_source(self, output_name: str) -> "LayerDependency | None":
        matches = [s for s in self.sources if output_name in s.layer]
        if len(matches) > 1:
            raise ValueError(
                f"{output_name!r} is ambiguous across sources:"
                f" {[s.layer.name for s in matches]!r} (layer {self.name!r})."
            )
        return matches[0] if matches else None


@dataclass(frozen=True, kw_only=True)
class DependencySpec:
    """A DependencySpec specifies the core attributes of an asset dependency."""

    output_name: str
    layer: Layer
    partition_mapping: dg.PartitionMapping | None = None
    metadata: Mapping[str, Any] | None = None
    parameter_name: InitVar[str | EllipsisType | None] = ...

    input_name: str | None = field(init=False)
    key: dg.AssetKey = field(init=False)

    def __post_init__(self, parameter_name: str | EllipsisType | None) -> None:
        # fail fast if ``output_name`` is not a member of ``layer``
        object.__setattr__(self, "key", self.layer.resolve_key(self.output_name))
        object.__setattr__(
            self,
            "input_name",
            self.output_name if parameter_name is ... else parameter_name,
        )

    def to_dep(self, metadata: Mapping[str, Any] | None = None) -> dg.AssetDep:
        return dg.AssetDep(
            asset=self.key,
            partition_mapping=self.partition_mapping,
            metadata={**(self.metadata or {}), **(metadata or {})},
        )

    def to_in(self, metadata: Mapping[str, Any] | None = None) -> dg.AssetIn:
        return dg.AssetIn(
            key=self.key,
            partition_mapping=self.partition_mapping,
            dagster_type=self.layer.python_type,
            metadata={**(self.metadata or {}), **(metadata or {})},
        )


@dataclass(frozen=True, kw_only=True)
class LayerDependency:
    layer: Layer
    partition_mapping: dg.PartitionMapping | None = None

    def resolve_dependency(
        self,
        output_name: str,
        input_name: str | None | EllipsisType = ...,
        metadata: Mapping[str, Any] | None = None,
    ) -> DependencySpec:
        return DependencySpec(
            output_name=output_name,
            layer=self.layer,
            parameter_name=input_name,
            partition_mapping=self.partition_mapping,
            metadata=metadata,
        )
