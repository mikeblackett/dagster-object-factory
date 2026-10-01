"""Components that build dagster assets from objects."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import dagster as dg

from dagster_object_factory.translator import DagsterObjectTranslator


@dataclass(kw_only=True, frozen=True)
class ObjectFactoryComponent[T, R](dg.Component, ABC):
    """A Component that builds one multi-asset per object via a translator.

    Subclasses implement ``execute`` to produce each object's output values.
    The translator supplies each asset's identity and dependencies; the
    component supplies its materialization behavior and shared attributes.

    Args:
        translator: Translates each object into its asset description.
        objects: Objects to build assets for.
        io_manager_key: IO manager key applied to all asset outs.
        deps: Deps merged into every built asset.
        metadata: Metadata merged into every built asset.
        owners: Owners merged into every built asset.
        tags: Tags merged into every built asset.
        kinds: Kinds merged into every built asset.
    """

    translator: DagsterObjectTranslator[T, R]
    objects: Sequence[T]

    io_manager_key: str | None = None
    deps: Iterable[dg.AssetDep] | None = None
    metadata: Mapping[str, Any] | None = None
    owners: Sequence[str] | None = None
    tags: Mapping[str, str] | None = None
    kinds: set[str] | None = None

    @abstractmethod
    def execute(
        self,
        context: dg.AssetExecutionContext,
        obj: T,
        ins: Mapping[str, Any],
        **kwargs,
    ) -> R | Sequence[R]:
        """Execute the object for a partition.

        Args:
            context: The asset execution context.
            obj: The object to execute.
            ins: Upstream inputs keyed by input name.
            **kwargs: Extra keyword arguments from ``resolve_execution_kwargs``.

        Returns:
            The output values. A single value for single-output assets, or a
            sequence aligned with the asset's output names for multi-output
            assets.
        """
        ...

    def resolve_execution_kwargs(
        self, context: dg.AssetExecutionContext, obj: T
    ) -> dict[str, Any]:
        """Resolve extra keyword arguments for ``execute``.

        Subclasses typically use this to derive arguments from the execution
        context, such as the current partition.

        Args:
            context: The asset execution context.
            obj: The object to execute.

        Returns:
            Keyword arguments passed to ``execute``. Defaults to an empty dict.
        """
        return {}

    def build_defs(self, context: dg.ComponentLoadContext) -> dg.Definitions:
        """Build the definitions of the component.

        Args:
            context: The component load context.

        Returns:
            A Definitions holding one multi-asset per object, with the
            component-level ``deps``, ``metadata``, ``owners``, ``tags``, and
            ``kinds`` merged into each.
        """
        assets = dg.map_asset_specs(
            lambda spec: spec.merge_attributes(
                deps=self.deps or ...,
                metadata=self.metadata or ...,
                owners=self.owners or ...,
                tags=self.tags or ...,
                kinds=self.kinds or ...,
            ),
            [self.make_asset(context, obj) for obj in self.objects],
        )
        return dg.Definitions(assets=assets)

    def make_asset(
        self,
        context: dg.ComponentLoadContext,
        obj: T,
    ) -> dg.AssetsDefinition:
        """Build the asset of a single object.

        Args:
            context: The component load context.
            obj: The object to build an asset for.

        Returns:
            A multi_asset whose outs, ins, and deps come from translating the
            object. At execution it calls ``execute`` and materializes one
            value per output, with metadata from ``get_result_metadata``.
        """
        translation = self.translator(obj)

        @dg.multi_asset(
            name=translation.name,
            description=translation.description,
            ins=translation.get_asset_ins(),
            deps=translation.get_extra_deps(),
            outs=translation.get_asset_outs(io_manager_key=self.io_manager_key),
            partitions_def=translation.partitions_def,
            required_resource_keys=self.get_required_resource_keys(),
        )
        def _asset(
            context: dg.AssetExecutionContext, **ins: Any
        ) -> Iterator[dg.MaterializeResult[R]]:
            kwargs = self.resolve_execution_kwargs(context, obj)
            result = self.execute(context, obj, ins, **kwargs)
            for output_name, value in translation.iter_output_values(result):
                yield dg.MaterializeResult(
                    value=value,
                    asset_key=translation.keys_by_output_name[output_name],
                    metadata=self.get_result_metadata(context, value),
                )

        return _asset

    def get_result_metadata(
        self, context: dg.AssetExecutionContext, value: R
    ) -> dict[str, dg.MetadataValue]:
        """Build the execution metadata of a materialized output value.

        Args:
            context: The asset execution context.
            value: The value being materialized.

        Returns:
            Metadata attached to each materialize result the built asset
            yields. Defaults to an empty dict.
        """
        return {}

    def get_required_resource_keys(self) -> frozenset[str]:
        """Return the resource keys required by the built assets.

        Returns:
            The required resource keys. Defaults to an empty set.
        """
        return frozenset()

    def get_partition_key(
        self, context: dg.AssetExecutionContext, dimension: str
    ) -> str:
        """Return the partition key of the execution context for a dimension.

        Args:
            context: The asset execution context.
            dimension: Dimension to read for multi-partition keys.

        Returns:
            The key of the given dimension for multi-partition keys, otherwise
            the scalar partition key.
        """
        key = context.partition_key
        if isinstance(key, dg.MultiPartitionKey):
            return key.keys_by_dimension[dimension]
        return key
