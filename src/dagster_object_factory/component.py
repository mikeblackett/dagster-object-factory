from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import dagster as dg

from dagster_object_factory.translator import DagsterObjectTranslator


@dataclass(kw_only=True, frozen=True)
class ObjectFactoryComponent[T, R](dg.Component, ABC):
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
    ) -> R: ...

    def resolve_execution_kwargs(
        self, context: dg.AssetExecutionContext, obj: T
    ) -> dict[str, Any]:
        return {}

    def build_defs(self, context: dg.ComponentLoadContext) -> dg.Definitions:
        assets = dg.map_asset_specs(
            lambda spec: spec.merge_attributes(
                deps=self.deps or ...,
                metadata=self.metadata or ...,
                owners=self.owners or ...,
                tags=self.tags or ...,
                kinds=self.kinds or ...,
            ),
            [self.make_asset(context, item) for item in self.objects],
        )
        return dg.Definitions(assets=assets)

    def make_asset(
        self,
        context: dg.ComponentLoadContext,
        obj: T,
    ) -> dg.AssetsDefinition:

        @dg.multi_asset(
            name=self.translator.get_asset_key(obj).to_python_identifier(),
            ins=self.translator.get_asset_ins(obj),
            deps=self.translator.get_extra_deps(obj),
            outs=self.translator.to_asset_outs(obj, io_manager_key=self.io_manager_key),
            required_resource_keys=self.get_required_resource_keys(),
            partitions_def=self.translator.get_partitions_def(obj),
        )
        def _asset(
            context: dg.AssetExecutionContext, **ins: Any
        ) -> dg.MaterializeResult[R]:
            kwargs = self.resolve_execution_kwargs(context, obj)
            result = self.execute(context, obj, ins, **kwargs)
            metadata = self.get_metadata(context, result)
            return dg.MaterializeResult(value=result, metadata=metadata)

        return _asset

    def get_metadata(
        self, context: dg.AssetExecutionContext, result: R
    ) -> dict[str, dg.MetadataValue]:
        return {}

    def get_required_resource_keys(self) -> frozenset[str]:
        return frozenset()

    def get_partition_key(
        self, context: dg.AssetExecutionContext, dimension: str
    ) -> str:
        key = context.partition_key
        if isinstance(key, dg.MultiPartitionKey):
            return key.keys_by_dimension[dimension]
        return key
