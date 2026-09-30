from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import dagster as dg

from dagster_object_factory.layer import DependencySpec, Layer


@dataclass(frozen=True, kw_only=True)
class DagsterObjectTranslation[T, R]:
    # attributes shared across outputs
    name: str
    output_names: tuple[str, ...]
    description: str | None
    dependency_specs: tuple[DependencySpec, ...]
    code_version: str | None
    partitions_def: dg.PartitionsDefinition | None
    group_name: str | None
    dagster_type: dg.DagsterType | type[R]
    # per-output attributes
    keys_by_output_name: Mapping[str, dg.AssetKey]
    descriptions_by_output_name: Mapping[str, str | None]
    tags_by_output_name: Mapping[str, Mapping[str, str] | None]
    metadata_by_output_name: Mapping[str, Mapping[str, dg.MetadataValue]]
    kinds_by_output_name: Mapping[str, set[str] | None]

    @property
    def keys(self) -> Sequence[dg.AssetKey]:
        return [self.keys_by_output_name[name] for name in self.output_names]

    @property
    def is_multi_output(self) -> bool:
        return len(self.output_names) > 1

    def get_asset_spec(self, output_name: str) -> dg.AssetSpec:
        return self._get_identity_spec(output_name=output_name).replace_attributes(
            deps=self.get_asset_deps()
        )

    def get_asset_out(
        self,
        output_name: str,
        *,
        io_manager_key: str | None = None,
    ) -> dg.AssetOut:
        return dg.AssetOut.from_spec(
            spec=self._get_identity_spec(output_name),
            io_manager_key=io_manager_key,
            dagster_type=self.dagster_type,
        )

    def get_asset_outs(
        self,
        *,
        io_manager_key: str | None = None,
    ) -> Mapping[str, dg.AssetOut]:
        return {
            output_name: self.get_asset_out(output_name, io_manager_key=io_manager_key)
            for output_name in self.output_names
        }

    def get_asset_deps(self) -> Iterable[dg.AssetDep]:
        return [d.to_dep() for d in self.dependency_specs]

    def get_extra_deps(self) -> Iterable[dg.AssetDep]:
        return [d.to_dep() for d in self.dependency_specs if d.input_name is None]

    def get_asset_ins(self) -> Mapping[str, dg.AssetIn]:
        return {
            d.input_name: d.to_in()
            for d in self.dependency_specs
            if d.input_name is not None
        }

    def iter_output_values(self, result: R | Iterable[R]) -> Iterator[tuple[str, R]]:
        values: Sequence[R]
        if self.is_multi_output:
            values = cast(Sequence[R], result)
        else:
            values = (cast(R, result),)
        return iter(zip(self.output_names, values, strict=True))

    def _get_identity_spec(self, output_name: str) -> dg.AssetSpec:
        return dg.AssetSpec(
            key=self.keys_by_output_name[output_name],
            description=self.descriptions_by_output_name[output_name],
            code_version=self.code_version,
            group_name=self.group_name,
            partitions_def=self.partitions_def,
            metadata=self.metadata_by_output_name[output_name],
            tags=self.tags_by_output_name[output_name],
            kinds=self.kinds_by_output_name[output_name],
        )


class DagsterObjectTranslator[T, R](ABC):
    layer: Layer[R]

    def __init__(self, layer: Layer[R]) -> None:
        self.layer = layer

    def __call__(self, obj: T) -> DagsterObjectTranslation[T, R]:
        return DagsterObjectTranslation(
            name=self.get_name(obj),
            description=self.get_description(obj),
            output_names=self.get_output_names(obj),
            keys_by_output_name=self.get_keys_by_output_name(obj),
            descriptions_by_output_name=self.get_descriptions_by_output_name(obj),
            tags_by_output_name=self.get_tags_by_output_name(obj),
            metadata_by_output_name=self.get_metadata_by_output_name(obj),
            dependency_specs=tuple(self.resolve_dependency_specs(obj)),
            kinds_by_output_name=self.get_kinds_by_output_name(obj),
            dagster_type=self.layer.python_type,
            partitions_def=self.get_partitions_def(obj),
            code_version=self.get_code_version(obj),
            group_name=self.get_group_name(obj),
        )

    @abstractmethod
    def resolve_dependency_specs(self, obj: T) -> Iterable[DependencySpec]: ...

    @abstractmethod
    def get_name(self, obj: T) -> str: ...

    def get_output_names(self, obj: T) -> tuple[str, ...]:
        return (self.get_name(obj),)

    def get_keys_by_output_name(self, obj: T) -> Mapping[str, dg.AssetKey]:
        return {
            output_name: self.layer.resolve_key(output_name)
            for output_name in self.get_output_names(obj)
        }

    def get_description(self, obj: T) -> str | None:
        return None

    def get_descriptions_by_output_name(self, obj: T) -> Mapping[str, str | None]:
        description = self.get_description(obj)
        return {output_name: description for output_name in self.get_output_names(obj)}

    def get_tags(self, obj: T) -> Mapping[str, str] | None:
        return {}

    def get_tags_by_output_name(self, obj: T) -> Mapping[str, Mapping[str, str] | None]:
        tags = self.get_tags(obj)
        return {output_name: tags for output_name in self.get_output_names(obj)}

    def get_kinds(self, obj: T) -> set[str] | None:
        return None

    def get_kinds_by_output_name(self, obj: T) -> Mapping[str, set[str] | None]:
        kinds = self.get_kinds(obj)
        return {output_name: kinds for output_name in self.get_output_names(obj)}

    def get_metadata(self, obj: T) -> Mapping[str, dg.MetadataValue]:
        return {}

    def get_metadata_by_output_name(
        self, obj: T
    ) -> Mapping[str, Mapping[str, dg.MetadataValue]]:
        metadata = self.get_metadata(obj)
        return {output_name: metadata for output_name in self.get_output_names(obj)}

    def get_group_name(self, obj: T) -> str | None:
        return self.layer.name

    def get_code_version(self, obj: T) -> str | None:
        return None

    def get_partitions_def(self, obj: T) -> dg.PartitionsDefinition | None:
        return self.layer.partitions_def
