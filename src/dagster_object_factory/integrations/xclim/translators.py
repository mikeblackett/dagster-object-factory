from collections.abc import Generator, Mapping
from typing import Final, cast

import dagster as dg
import xarray as xr
import xclim.core.indicator as xc

from dagster_object_factory.layer import DependencySpec
from dagster_object_factory.translator import DagsterObjectTranslator

DAGSTER_XCLIM_ASSET_KINDS: Final = {
    xc.InputKind.VARIABLE,
    xc.InputKind.OPTIONAL_VARIABLE,
}
DEFAULT_XCLIM_KINDS: Final = {"xarray"}

DEFAULT_KEYWORD_SEPARATOR: Final = None
DEFAULT_SRC_FREQ: Final = "D"


class DagsterXclimIndicatorTranslator[T: xc.Indicator](
    DagsterObjectTranslator[T, xr.DataArray]
):
    def get_name(self, obj: T) -> str:
        return cast(str, obj.identifier)

    def get_output_names(self, obj: T) -> tuple[str, ...]:
        names = tuple(output["var_name"] for output in obj.cf_attrs)
        return names or (self.get_name(obj),)

    def get_description(self, obj: T) -> str | None:
        if obj.title is None:
            return obj.abstract
        description = obj.title
        if obj.abstract:
            description += f"\n\n{obj.abstract}"
        return description

    def get_descriptions_by_output_name(self, obj: T) -> Mapping[str, str | None]:
        if obj.n_outs == 1:
            return super().get_descriptions_by_output_name(obj)
        values = [
            (output["var_name"], output["description"]) for output in obj.cf_attrs
        ]
        return dict(values)

    def resolve_dependency_specs(self, obj: T) -> Generator[DependencySpec]:
        parameters = cast(dict[str, xc.Parameter], obj.parameters)
        for name, parameter in parameters.items():
            if parameter.kind not in DAGSTER_XCLIM_ASSET_KINDS:
                continue
            canonical_name = _resolve_canonical_parameter_name(parameter)
            source = self.layer.resolve_source(canonical_name)
            if source is None:
                if parameter.kind is xc.InputKind.OPTIONAL_VARIABLE:
                    continue
                raise dg.DagsterInvalidDefinitionError(
                    f"{obj.identifier}: parameter {name!r} resolves to unknown "
                    f"variable {canonical_name!r}"
                )
            yield source.resolve_dependency(output_name=canonical_name, input_name=name)

    def get_metadata(self, obj: T) -> Mapping[str, dg.MetadataValue]:
        metadata: dict[str, dg.MetadataValue] = {
            "indice": dg.MetadataValue.python_artifact(obj.compute),
            "obj": dg.MetadataValue.python_artifact(obj.__class__),
            "src_freq": dg.MetadataValue.text(
                getattr(obj, "src_freq", DEFAULT_SRC_FREQ)
            ),
        }
        if obj.injected_parameters:
            metadata["injected_parameters"] = dg.MetadataValue.json(
                obj.injected_parameters
            )
        return metadata

    def get_tags(self, obj: T) -> Mapping[str, str]:
        return {k: "true" for k in obj.keywords.split(DEFAULT_KEYWORD_SEPARATOR)}

    def get_kinds(self, obj: T) -> set[str] | None:
        return DEFAULT_XCLIM_KINDS


def _resolve_canonical_parameter_name(parameter: xc.Parameter) -> str:
    # handles parameter name overrides in virtual modules
    if isinstance(parameter.default, str):
        return parameter.default
    return parameter.compute_name
