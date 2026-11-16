"""Hypothesis strategies for testing dagster-object-factory definitions.

These strategies generate the library's building blocks — names, keys,
layers, and dependency specs — for use in property-based tests, both here
and in downstream projects that extend the library.
"""

import re
from collections.abc import Mapping, Sequence
from types import EllipsisType
from typing import Any

import dagster as dg
import hypothesis.strategies as st

from dagster_object_factory.layer import DependencySpec, Layer, LayerDep

DAGSTER_NAME_REGEX = re.compile(r"^[A-Za-z0-9_]+$")
PYTHON_TYPES: list[type] = [str, int, float, list, dict]


@st.composite
def names(draw: st.DrawFn) -> str:
    """A valid dagster identifier."""
    return draw(st.from_regex(DAGSTER_NAME_REGEX, fullmatch=True))


@st.composite
def key_prefixes(
    draw: st.DrawFn, min_size: int = 0, max_size: int = 2
) -> Sequence[str]:
    """A sequence of unique name parts, sometimes a list, sometimes a tuple."""
    parts = draw(st.lists(names(), min_size=min_size, max_size=max_size, unique=True))
    if draw(st.booleans()):
        return tuple(parts)
    return parts


@st.composite
def python_types(draw: st.DrawFn, exclude: Sequence[type] = ()) -> type:
    """A python type drawn from ``PYTHON_TYPES``, excluding ``exclude``."""
    elements = [t for t in PYTHON_TYPES if t not in exclude]
    return draw(st.sampled_from(elements))


@st.composite
def asset_keys(
    draw: st.DrawFn,
    name: st.SearchStrategy[str] | None = None,
    key_prefix: st.SearchStrategy[Sequence[str]] | None = None,
) -> dg.AssetKey:
    """An asset key built from a name and an optional prefix."""
    if name is None:
        name = names()
    if key_prefix is None:
        key_prefix = key_prefixes()
    return dg.AssetKey(draw(name)).with_prefix(draw(key_prefix))


@st.composite
def metadata(
    draw: st.DrawFn, values: st.SearchStrategy | None = None
) -> Mapping[str, Any] | None:
    if values is None:
        values = st.one_of([st.text(), st.integers(), st.booleans()])
    return draw(
        st.one_of(
            st.none(),
            st.dictionaries(names(), values),
        )
    )


@st.composite
def layers(
    draw: st.DrawFn,
    *,
    name: str | st.SearchStrategy[str] | None = None,
    min_outputs: int = 1,
    max_outputs: int = 3,
    output_names: st.SearchStrategy[frozenset[str]] | None = None,
    key_prefix: st.SearchStrategy[Sequence[str]] | None = None,
    python_type: type | st.SearchStrategy[type] | None = None,
    layer_deps: st.SearchStrategy[Sequence[Layer]] | None = None,
) -> Layer:
    """A layer with generated identity attributes.

    Args:
        name: Name or name strategy. Defaults to a generated name.
        min_outputs: Minimum number of declared outputs.
        max_outputs: Maximum number of declared outputs.
        output_names: Output names strategy. Defaults to generated unique
            names within the ``min_outputs``/``max_outputs`` bounds.
        key_prefix: Key prefix strategy. Defaults to a short generated prefix.
        python_type: Output type or type strategy. Defaults to a generated
            python type.
        layer_deps: Upstream layer strategy. Defaults to no layer deps.
    """
    name_strategy: st.SearchStrategy[str]
    if isinstance(name, st.SearchStrategy):
        name_strategy = name
    else:
        name_strategy = st.just(name) if name is not None else names()
    if output_names is None:
        output_names = st.lists(
            names(), min_size=min_outputs, max_size=max_outputs, unique=True
        ).map(frozenset)
    if key_prefix is None:
        key_prefix = key_prefixes()
    if python_type is None or isinstance(python_type, st.SearchStrategy):
        type_strategy: st.SearchStrategy[type] = (
            python_types() if python_type is None else python_type
        )
    else:
        type_strategy = st.just(python_type)
    if layer_deps is None:
        layer_deps = st.just(())
    return Layer(
        name=draw(name_strategy),
        key_prefix=tuple(draw(key_prefix)),
        output_names=draw(output_names),
        python_type=draw(type_strategy),
        layer_deps=tuple(LayerDep(layer=layer) for layer in draw(layer_deps)),
    )


@st.composite
def dependency_specs(
    draw: st.DrawFn,
    layer: Layer,
    *,
    input_names: st.SearchStrategy[str | EllipsisType | None] | None = None,
    metadata: st.SearchStrategy[Mapping[str, Any] | None] | None = None,
    partition_mapping: st.SearchStrategy[dg.PartitionMapping | None] | None = None,
) -> DependencySpec:
    """Dependency specs on the declared outputs of a layer.

    Args:
        layer: The upstream layer the specs depend on.
        min_specs: Minimum number of specs.
        max_specs: Maximum number of specs. Defaults to the number of declared
            outputs.
        input_names: Input name strategy. Defaults to a random mix of Ellipsis
            (reuse the output name), a generated name, and None (input-less).
        metadata: Metadata strategy. Defaults to None or a small string mapping.
        partition_mapping: Partition mapping strategy. Defaults to None.
    """
    output_name = draw(st.sampled_from(sorted(layer.output_names)))
    if input_names is None:
        input_names = st.one_of(st.none(), st.just(Ellipsis), names())
    if metadata is None:
        metadata = st.one_of(st.none(), st.dictionaries(names(), st.text()))
    if partition_mapping is None:
        partition_mapping = st.none()
    return DependencySpec(
        output_name=output_name,
        layer=layer,
        input_name=draw(input_names),
        metadata=draw(metadata),
        partition_mapping=draw(partition_mapping),
    )
