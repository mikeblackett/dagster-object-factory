from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import dagster as dg

from dagster_object_factory.layer import DependencySpec, Layer


@dataclass(frozen=True, kw_only=True)
class DagsterObjectTranslator[T, R](ABC):
    layer: Layer[R]

    @abstractmethod
    def resolve_dependency_specs(self, obj: T) -> Iterable[DependencySpec]: ...

    @abstractmethod
    def get_name(self, obj: T) -> str: ...

    def _to_identity_spec(self, obj: T) -> dg.AssetSpec:
        # Identity-only spec to avoid unnecessary dependency resolution trips
        return dg.AssetSpec(
            key=self.get_asset_key(obj),
            description=self.get_description(obj),
            code_version=self.get_code_version(obj),
            group_name=self.get_group_name(obj),
            partitions_def=self.get_partitions_def(obj),
            metadata=self.get_metadata(obj),
            tags=self.get_tags(obj),
            kinds=self.get_kinds(obj),
        )

    def to_asset_spec(self, obj: T) -> dg.AssetSpec:
        return self._to_identity_spec(obj).replace_attributes(
            deps=self.get_asset_deps(obj)
        )

    def to_asset_out(self, obj: T, *, io_manager_key: str | None = None) -> dg.AssetOut:
        return dg.AssetOut.from_spec(
            self._to_identity_spec(obj),
            io_manager_key=io_manager_key,
            dagster_type=self.layer.python_type,
        )

    def to_asset_outs(
        self,
        obj: T,
        *,
        io_manager_key: str | None = None,
    ) -> dict[str, dg.AssetOut]:
        return {
            self.get_asset_key(obj).to_python_identifier(): self.to_asset_out(
                obj,
                io_manager_key=io_manager_key,
            )
        }

    def get_asset_deps(self, obj: T) -> Iterable[dg.AssetDep]:
        return [d.to_dep() for d in self.resolve_dependency_specs(obj)]

    def get_extra_deps(self, obj: T) -> Iterable[dg.AssetDep]:
        return [
            d.to_dep()
            for d in self.resolve_dependency_specs(obj)
            if d.input_name is None
        ]

    def get_asset_ins(self, obj: T) -> Mapping[str, dg.AssetIn]:
        return {
            d.input_name: d.to_in()
            for d in self.resolve_dependency_specs(obj)
            if d.input_name is not None
        }

    def get_asset_key(self, obj: T) -> dg.AssetKey:
        return self.layer.resolve_key(self.get_name(obj))

    def get_group_name(self, obj: T) -> str | None:
        return self.layer.name

    def get_code_version(self, obj: T) -> str | None:
        return None

    def get_description(self, obj: T) -> str | None:
        return None

    def get_metadata(self, obj: T) -> Mapping[str, dg.MetadataValue]:
        return {}

    def get_tags(self, obj: T) -> Mapping[str, str]:
        return {}

    def get_kinds(self, obj: T) -> set[str] | None:
        return None

    def get_partitions_def(self, obj: T) -> dg.PartitionsDefinition | None:
        return self.layer.partitions_def
