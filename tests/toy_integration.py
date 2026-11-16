from collections.abc import Iterable, Mapping
from typing import Any

import dagster as dg

from dagster_object_factory import (
    DagsterObjectTranslator,
    DependencySpec,
    ObjectFactoryComponent,
)

from .toy_example import Collection, Composite, Leaf


class LeafTranslator(DagsterObjectTranslator[Leaf, str]):
    def get_name(self, obj: Leaf, /) -> str:
        return obj.name

    def resolve_dependency_specs(self, obj: Leaf, /) -> Iterable[DependencySpec]:
        return ()


class CompositeTranslator(DagsterObjectTranslator[Composite, str]):
    def get_name(self, obj: Composite, /) -> str:
        return obj.name

    def resolve_dependency_specs(self, obj: Composite, /) -> Iterable[DependencySpec]:
        for part in obj.parts:
            layer_dep = self.layer.resolve_layer_dep(part.name)
            if layer_dep is None:
                raise ValueError(f"{part.name!r} is not a part of {obj.name!r}")
            yield layer_dep.resolve_dependency(part.name)


class CollectionTranslator(DagsterObjectTranslator[Collection, str]):
    def get_name(self, obj: Collection, /) -> str:
        return obj.name

    def get_output_names(self, obj: Collection, /) -> tuple[str, ...]:
        return tuple(member.name for member in obj.members)

    def resolve_dependency_specs(self, obj: Collection, /) -> Iterable[DependencySpec]:
        for member in obj.members:
            layer_dep = self.layer.resolve_layer_dep(member.name)
            if layer_dep is None:
                raise ValueError(f"{member.name!r} is not a member of {obj.name!r}")
            yield layer_dep.resolve_dependency(member.name)


class LeafComponent(ObjectFactoryComponent[Leaf, str]):
    def execute(
        self,
        context: dg.AssetExecutionContext,
        obj: Leaf,
        ins: Mapping[str, Any],
        **kwargs: Any,
    ) -> str:
        return obj.value


class CompositeComponent(ObjectFactoryComponent[Composite, str]):
    def execute(
        self,
        context: dg.AssetExecutionContext,
        obj: Composite,
        ins: Mapping[str, Any],
        **kwargs: Any,
    ) -> str:
        return obj.value()


class CollectionComponent(ObjectFactoryComponent[Collection, str]):
    def execute(
        self,
        context: dg.AssetExecutionContext,
        obj: Collection,
        ins: Mapping[str, Any],
        **kwargs: Any,
    ) -> tuple[str, ...]:
        return obj.value()
