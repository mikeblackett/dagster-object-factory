"""Asset layers and the dependencies between them."""

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from types import EllipsisType
from typing import Any

import dagster as dg


@dataclass(frozen=True, kw_only=True, eq=False)
class Layer[R]:
    """A group of assets with a shared key space, output type, and dependency sources.

    A translator maps each object to a multi-asset whose outputs must be
    declared up front in ``output_names``; resolving an undeclared output
    raises.

    Args:
        name: Name of the layer, used as the default asset ``group_name``.
        description: Description of the layer.
        key_prefix: Prefix applied to every asset key in the layer.
        output_names: Names of the outputs the layer produces.
        python_type: Python type shared by all of the layer's outputs.
        partitions_def: Partitions definition applied to the layer's assets.
        sources: Upstream layers this layer depends on.
    """

    name: str
    description: str | None = None
    key_prefix: tuple[str, ...]
    output_names: frozenset[str]
    python_type: type[R]
    partitions_def: dg.PartitionsDefinition | None = None
    sources: "tuple[LayerDependency, ...]" = field(default_factory=tuple)

    # TODO: Rename `sources` to `layer_dependencies` or `layer_deps`?
    def __contains__(self, output_name: str) -> bool:
        """Return whether ``output_name`` is a declared output of the layer."""
        return output_name in self.output_names

    def resolve_key(self, output_name: str) -> dg.AssetKey:
        """Resolve the asset key of a declared output.

        Args:
            output_name: Name of the output.

        Returns:
            The asset key of the output, i.e. the output name prefixed with
            the layer's key prefix.

        Raises:
            ValueError: If the output is not declared by the layer.
        """
        if output_name not in self:
            raise ValueError(
                f"{output_name!r} is not a declared output of layer {self.name!r}."
            )
        return dg.AssetKey(output_name).with_prefix(self.key_prefix)

    def resolve_source(self, output_name: str) -> "LayerDependency | None":
        """Find the source that produces an output.

        Args:
            output_name: Name of the output.

        Returns:
            The layer dependency that produces the output, or None if none does.

        Raises:
            ValueError: If more than one source produces the output.
        """
        # TODO: Rename this method `find_source`? This would make a distinction between resolving (can't fail) and finding (can fail)
        matches = [s for s in self.sources if output_name in s.layer]
        if len(matches) > 1:
            raise ValueError(
                f"{output_name!r} is ambiguous across sources:"
                f" {[s.layer.name for s in matches]!r} (layer {self.name!r})."
            )
        return matches[0] if matches else None


@dataclass(frozen=True, kw_only=True)
class DependencySpec:
    """Specifies a dependency on an output of a source layer.

    A spec renders as an ``AssetDep`` via ``to_dep``, which records the
    dependency without passing a value, or as an ``AssetIn`` via ``to_in``,
    which also passes the upstream value to a named input.

    Args:
        output_name: Name of the upstream output.
        layer: Layer the upstream output belongs to.
        partition_mapping: Partition mapping applied to the dependency.
        metadata: Metadata attached to the dependency.
        parameter_name: Name of the asset input the dependency feeds. Ellipsis
            (the default) reuses ``output_name``; None makes the dependency
            input-less, so it is emitted as an ``AssetDep`` rather than an
            ``AssetIn``.

    Attributes:
        input_name: Resolved input name; None for input-less dependencies.
        key: Asset key of the upstream output.
    """

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
        """Build the dependency as an asset dep.

        Args:
            metadata: Metadata merged over the spec's metadata.

        Returns:
            An AssetDep on the upstream key, carrying the spec's partition
            mapping and metadata, with the given metadata taking precedence.
        """
        return dg.AssetDep(
            asset=self.key,
            partition_mapping=self.partition_mapping,
            metadata={**(self.metadata or {}), **(metadata or {})},
        )

    def to_in(self, metadata: Mapping[str, Any] | None = None) -> dg.AssetIn:
        """Build the dependency as an asset input.

        Args:
            metadata: Metadata merged over the spec's metadata.

        Returns:
            An AssetIn on the upstream key, typed with the layer's python
            type, carrying the spec's partition mapping and metadata, with
            the given metadata taking precedence.
        """
        return dg.AssetIn(
            key=self.key,
            partition_mapping=self.partition_mapping,
            dagster_type=self.layer.python_type,
            metadata={**(self.metadata or {}), **(metadata or {})},
        )


@dataclass(frozen=True, kw_only=True)
class LayerDependency:
    """A source layer whose outputs the layer's assets may depend on, along
    with the partition mapping shared by every dependency resolved from it.

    Args:
        layer: The upstream layer.
        partition_mapping: Partition mapping applied to every dependency
            resolved from the layer.
    """

    # TODO: Rename `LayerDependency` to `LayerDep` to better match dagster style e.g. `AssetDep`?
    layer: Layer
    partition_mapping: dg.PartitionMapping | None = None

    def resolve_dependency(
        self,
        output_name: str,
        input_name: str | None | EllipsisType = ...,
        metadata: Mapping[str, Any] | None = None,
    ) -> DependencySpec:
        """Resolve a dependency on one of the layer's outputs.

        Args:
            output_name: Name of the upstream output.
            input_name: Name of the asset input the dependency feeds. Ellipsis
                (the default) reuses ``output_name``; None makes the dependency
                input-less.
            metadata: Metadata attached to the dependency.

        Returns:
            A DependencySpec on the asset key of the layer's output.
        """
        return DependencySpec(
            output_name=output_name,
            layer=self.layer,
            parameter_name=input_name,
            partition_mapping=self.partition_mapping,
            metadata=metadata,
        )
