"""Translation of objects into dagster assets."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import dagster as dg

from dagster_object_factory.layer import DependencySpec, Layer


@dataclass(frozen=True, kw_only=True)
class DagsterObjectTranslation[T, R]:
    """An immutable description of how an object maps to a dagster asset.

    Produced by a ``DagsterObjectTranslator`` and consumed by
    ``ObjectFactoryComponent.make_asset`` to build the object's multi-asset.
    Attributes without a ``_by_output_name`` suffix are shared across the
    asset's outputs; the rest are keyed by output name.

    Args:
        name: Name of the asset.
        output_names: Names of the asset's outputs, in materialization order.
        description: Description of the asset.
        dependency_specs: Dependencies on upstream assets.
        code_version: Code version shared by all outputs.
        partitions_def: Partitions definition shared by all outputs.
        group_name: A group name for the outputs of this asset.
        dagster_type: Python type of the outputs.
        keys_by_output_name: Asset key of each output.
        descriptions_by_output_name: Description of each output.
        tags_by_output_name: Tags of each output.
        metadata_by_output_name: Metadata of each output.
        kinds_by_output_name: Kinds of each output.
    """

    name: str
    output_names: tuple[str, ...]
    description: str | None
    dependency_specs: tuple[DependencySpec, ...]
    code_version: str | None
    partitions_def: dg.PartitionsDefinition | None
    group_name: str | None
    dagster_type: dg.DagsterType | type[R]
    keys_by_output_name: Mapping[str, dg.AssetKey]
    descriptions_by_output_name: Mapping[str, str | None]
    tags_by_output_name: Mapping[str, Mapping[str, str] | None]
    metadata_by_output_name: Mapping[str, Mapping[str, dg.MetadataValue]]
    kinds_by_output_name: Mapping[str, set[str] | None]

    @property
    def keys(self) -> Sequence[dg.AssetKey]:
        """Asset keys of all outputs, in output order."""
        return [self.keys_by_output_name[name] for name in self.output_names]

    @property
    def is_multi_output(self) -> bool:
        """Whether the asset has more than one output."""
        return len(self.output_names) > 1

    def get_asset_spec(self, output_name: str) -> dg.AssetSpec:
        """Build the asset spec of a single output.

        Args:
            output_name: Name of the output.

        Returns:
            The output's spec, with deps set to all dependency specs.
        """
        return self._get_identity_spec(output_name=output_name).replace_attributes(
            deps=self.get_asset_deps()
        )

    def get_asset_out(
        self,
        output_name: str,
        *,
        io_manager_key: str | None = None,
    ) -> dg.AssetOut:
        """Build the asset out of a single output.

        Args:
            output_name: Name of the output.
            io_manager_key: IO manager key of the out.

        Returns:
            An AssetOut for the output, typed with the translation's python
            type.
        """
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
        """Build the asset outs of all outputs.

        Args:
            io_manager_key: IO manager key of all outs.

        Returns:
            A mapping of output name to AssetOut.
        """
        return {
            output_name: self.get_asset_out(output_name, io_manager_key=io_manager_key)
            for output_name in self.output_names
        }

    def get_asset_deps(self) -> Iterable[dg.AssetDep]:
        """Build an asset dep for every dependency spec.

        Returns:
            An AssetDep for each spec: the asset's complete set of upstream
            dependencies, as attached by ``get_asset_spec``.
        """
        return [d.to_dep() for d in self.dependency_specs]

    def get_extra_deps(self) -> Iterable[dg.AssetDep]:
        """Build the asset deps of the input-less dependency specs.

        Returns:
            An AssetDep for each spec whose input name is None. These are
            attached to the asset as plain deps, since they do not feed a
            value into an input.
        """
        return [d.to_dep() for d in self.dependency_specs if d.input_name is None]

    def get_asset_ins(self) -> Mapping[str, dg.AssetIn]:
        """Build the asset inputs of the dependency specs.

        Returns:
            A mapping of input name to AssetIn, one per spec that has an
            input name. Input-less specs are excluded; see
            ``get_extra_deps``.
        """
        return {
            d.input_name: d.to_in()
            for d in self.dependency_specs
            if d.input_name is not None
        }

    def iter_output_values(self, result: R | Iterable[R]) -> Iterator[tuple[str, R]]:
        """Pair each output name with the value produced for it.

        Args:
            result: The asset's result. A single value for single-output
                translations, or a sequence aligned with the output names for
                multi-output translations.

        Yields:
            (output_name, value) pairs in output order.

        Raises:
            ValueError: If a multi-output result does not align with the
                output names.
        """
        values: Sequence[R]
        if self.is_multi_output:
            values = cast(Sequence[R], result)
        else:
            values = (cast(R, result),)
        return iter(zip(self.output_names, values, strict=True))

    def _get_identity_spec(self, output_name: str) -> dg.AssetSpec:
        """Build the AssetSpec of an output as declared by the translation.

        The spec carries the output's identity attributes but no deps;
        ``get_asset_spec`` and ``get_asset_out`` build on it.
        """
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
    """Translates objects into DagsterObjectTranslations for a layer.

    Subclasses implement ``resolve_dependency_specs`` and ``get_name`` and may
    override the remaining hooks to provide custom translation logic. Each
    per-object hook such as ``get_description`` has a corresponding
    ``*_by_output_name`` hook that defaults to broadcasting the per-object
    value to every output.

    Attributes:
        layer: Layer the translator produces assets for.
    """

    layer: Layer[R]

    def __init__(self, layer: Layer[R]) -> None:
        """Initialize the translator with the layer it translates for."""
        self.layer = layer

    def __call__(self, obj: T) -> DagsterObjectTranslation[T, R]:
        """Translate an object into a DagsterObjectTranslation.

        Args:
            obj: The object to translate.

        Returns:
            A translation built from every hook of the translator.
        """
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
    def resolve_dependency_specs(self, obj: T) -> Iterable[DependencySpec]:
        """Resolve the object's dependencies on the layer's source outputs.

        Args:
            obj: The object to translate.

        Returns:
            The dependency specs of the object.
        """
        ...

    @abstractmethod
    def get_name(self, obj: T) -> str:
        """Return the asset name of the object.

        Args:
            obj: The object to translate.

        Returns:
            The asset name.
        """
        ...

    def get_output_names(self, obj: T) -> tuple[str, ...]:
        """Return the output names of the object.

        Args:
            obj: The object to translate.

        Returns:
            The output names. Defaults to a single output named after the
            asset.
        """
        return (self.get_name(obj),)

    def get_keys_by_output_name(self, obj: T) -> Mapping[str, dg.AssetKey]:
        """Return the asset key of each output of the object.

        Args:
            obj: The object to translate.

        Returns:
            A mapping of output name to key, resolved from the layer.
        """
        return {
            output_name: self.layer.resolve_key(output_name)
            for output_name in self.get_output_names(obj)
        }

    def get_description(self, obj: T) -> str | None:
        """Return the asset description of the object.

        Args:
            obj: The object to translate.

        Returns:
            The description. Defaults to None.
        """
        return None

    def get_descriptions_by_output_name(self, obj: T) -> Mapping[str, str | None]:
        """Return the description of each output of the object.

        Args:
            obj: The object to translate.

        Returns:
            The shared description from ``get_description`` broadcast to every
            output.
        """
        description = self.get_description(obj)
        return {output_name: description for output_name in self.get_output_names(obj)}

    def get_tags(self, obj: T) -> Mapping[str, str] | None:
        """Return the tags of the object.

        Args:
            obj: The object to translate.

        Returns:
            The tags. Defaults to an empty mapping.
        """
        return {}

    def get_tags_by_output_name(self, obj: T) -> Mapping[str, Mapping[str, str] | None]:
        """Return the tags of each output of the object.

        Args:
            obj: The object to translate.

        Returns:
            The shared tags from ``get_tags`` broadcast to every output.
        """
        tags = self.get_tags(obj)
        return {output_name: tags for output_name in self.get_output_names(obj)}

    def get_kinds(self, obj: T) -> set[str] | None:
        """Return the kinds of the object.

        Args:
            obj: The object to translate.

        Returns:
            The asset kinds. Defaults to None.
        """
        return None

    def get_kinds_by_output_name(self, obj: T) -> Mapping[str, set[str] | None]:
        """Return the kinds of each output of the object.

        Args:
            obj: The object to translate.

        Returns:
            The shared kinds from ``get_kinds`` broadcast to every output.
        """
        kinds = self.get_kinds(obj)
        return {output_name: kinds for output_name in self.get_output_names(obj)}

    def get_metadata(self, obj: T) -> Mapping[str, dg.MetadataValue]:
        """Return the metadata of the object.

        Args:
            obj: The object to translate.

        Returns:
            The metadata. Defaults to an empty mapping.
        """
        return {}

    def get_metadata_by_output_name(
        self, obj: T
    ) -> Mapping[str, Mapping[str, dg.MetadataValue]]:
        """Return the metadata of each output of the object.

        Args:
            obj: The object to translate.

        Returns:
            The shared metadata from ``get_metadata`` broadcast to every
            output.
        """
        metadata = self.get_metadata(obj)
        return {output_name: metadata for output_name in self.get_output_names(obj)}

    def get_group_name(self, obj: T) -> str | None:
        """Return the asset group of the object.

        Args:
            obj: The object to translate.

        Returns:
            The group name. Defaults to the layer name.
        """
        return self.layer.name

    def get_code_version(self, obj: T) -> str | None:
        """Return the code version of the object.

        Args:
            obj: The object to translate.

        Returns:
            The code version. Defaults to None.
        """
        return None

    def get_partitions_def(self, obj: T) -> dg.PartitionsDefinition | None:
        """Return the partitions definition of the object.

        Args:
            obj: The object to translate.

        Returns:
            The partitions definition. Defaults to the layer's.
        """
        return self.layer.partitions_def
