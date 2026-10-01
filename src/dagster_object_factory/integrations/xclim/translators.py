"""Translation of xclim indicators into dagster assets."""

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
    """A translator mapping xclim indicators to dagster assets.

    The indicator's identifier becomes the asset name, its ``cf_attrs``
    become the asset's outputs, and its variable parameters become
    dependencies resolved against the layer's sources.
    """

    def get_name(self, obj: T) -> str:
        """Return the asset name of the indicator.

        Args:
            obj: The indicator to translate.

        Returns:
            The indicator's identifier.
        """
        return cast(str, obj.identifier)

    def get_output_names(self, obj: T) -> tuple[str, ...]:
        """Return the output names of the indicator.

        Args:
            obj: The indicator to translate.

        Returns:
            The var_name of each cf_attrs entry, or the asset name if the
            indicator declares none.
        """
        names = tuple(output["var_name"] for output in obj.cf_attrs)
        return names or (self.get_name(obj),)

    def get_description(self, obj: T) -> str | None:
        """Return the asset description of the indicator.

        Args:
            obj: The indicator to translate.

        Returns:
            The title with the abstract appended, or the abstract alone if
            the title is missing.
        """
        if obj.title is None:
            return obj.abstract
        description = obj.title
        if obj.abstract:
            description += f"\n\n{obj.abstract}"
        return description

    def get_descriptions_by_output_name(self, obj: T) -> Mapping[str, str | None]:
        """Return the description of each output of the indicator.

        Args:
            obj: The indicator to translate.

        Returns:
            The description of each cf_attrs entry for multi-output
            indicators, otherwise the shared description broadcast to every
            output.
        """
        if obj.n_outs == 1:
            return super().get_descriptions_by_output_name(obj)
        values = [
            (output["var_name"], output["description"]) for output in obj.cf_attrs
        ]
        return dict(values)

    def resolve_dependency_specs(self, obj: T) -> Generator[DependencySpec]:
        """Resolve the indicator's variable parameters as layer dependencies.

        Args:
            obj: The indicator to translate.

        Yields:
            A DependencySpec for each parameter of a variable input kind,
            named after the parameter and resolved against the layer's
            sources. Optional variables without a matching source are
            skipped.

        Raises:
            dagster.DagsterInvalidDefinitionError: If a required variable
                parameter resolves to an output of no source layer.
        """
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
        """Return the metadata of the indicator.

        Args:
            obj: The indicator to translate.

        Returns:
            Entries for the compute function (``indice``), the indicator
            class (``obj``), and the source frequency (``src_freq``), plus
            the indicator's injected parameters (``injected_parameters``) if
            it has any.
        """
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
        """Return the tags of the indicator.

        Args:
            obj: The indicator to translate.

        Returns:
            One tag per whitespace-separated keyword, each set to "true".
        """
        return {k: "true" for k in obj.keywords.split(DEFAULT_KEYWORD_SEPARATOR)}

    def get_kinds(self, obj: T) -> set[str] | None:
        """Return the kinds of the indicator.

        Args:
            obj: The indicator to translate.

        Returns:
            The default kinds for xclim indicator assets
            (``DEFAULT_XCLIM_KINDS``).
        """
        return DEFAULT_XCLIM_KINDS


def _resolve_canonical_parameter_name(parameter: xc.Parameter) -> str:
    """Resolve the canonical variable name of a parameter.

    Args:
        parameter: The indicator parameter.

    Returns:
        The parameter's string default if it has one, otherwise its compute
        name.
    """
    # handles parameter name overrides in virtual modules
    if isinstance(parameter.default, str):
        return parameter.default
    return parameter.compute_name
