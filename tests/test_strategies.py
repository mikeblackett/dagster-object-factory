from types import EllipsisType

import dagster as dg
import hypothesis as hp
import hypothesis.strategies as st

from dagster_object_factory.layer import Layer

from .strategies import (
    DAGSTER_NAME_REGEX,
    PYTHON_TYPES,
    asset_keys,
    dependency_specs,
    key_prefixes,
    layers,
    names,
    python_types,
)


def _assert_name(name: str) -> None:
    assert DAGSTER_NAME_REGEX.fullmatch(name)


@hp.given(name=names())
def test_names_are_dagster_identifiers(name: str) -> None:
    _assert_name(name)


@hp.given(prefix=key_prefixes(min_size=1, max_size=2))
def test_key_prefixes(prefix: list[str] | tuple[str, ...]) -> None:
    assert isinstance(prefix, (list, tuple))
    assert 1 <= len(prefix) <= 2
    assert len(set(prefix)) == len(prefix)
    for part in prefix:
        _assert_name(part)


@hp.given(dtype=python_types(exclude=[str, int]))
def test_python_types_respect_exclusion(dtype: type) -> None:
    assert dtype in PYTHON_TYPES
    assert dtype not in (str, int)


@hp.given(key=asset_keys())
def test_asset_keys(key: dg.AssetKey) -> None:
    assert key.path
    for part in key.path:
        _assert_name(part)


@hp.given(layer=layers())
def test_layers(layer: Layer) -> None:
    assert 1 <= len(layer.output_names) <= 3
    for output_name in layer.output_names:
        _assert_name(output_name)
    for part in layer.key_prefix:
        _assert_name(part)
    assert layer.python_type in PYTHON_TYPES
    assert layer.layer_deps == ()
    _assert_name(layer.name)


@hp.given(data=st.data())
def test_layers_respect_overrides(data: st.DataObject) -> None:
    outputs = data.draw(st.lists(names(), min_size=2, max_size=2, unique=True))
    prefix = data.draw(key_prefixes())
    layer = data.draw(
        layers(
            name="fixed_name",
            output_names=st.just(frozenset(outputs)),
            key_prefix=st.just(tuple(prefix)),
            python_type=int,
        )
    )
    assert layer.name == "fixed_name"
    assert layer.output_names == frozenset(outputs)
    assert layer.key_prefix == tuple(prefix)
    assert layer.python_type is int
    assert layer.layer_deps == ()


@hp.given(data=st.data())
def test_layers_layer_deps_wrapped(data: st.DataObject) -> None:
    upstream = data.draw(layers())
    layer = data.draw(layers(layer_deps=st.just((upstream,))))
    assert len(layer.layer_deps) == 1
    assert layer.layer_deps[0].layer is upstream
    assert layer.resolve_layer_dep(next(iter(upstream.output_names))) is not None


@hp.given(data=st.data())
def test_dependency_specs(data: st.DataObject) -> None:
    layer = data.draw(layers())
    spec = data.draw(dependency_specs(layer))
    assert spec.layer is layer
    assert spec.output_name in layer
    assert spec.key == layer.resolve_key(spec.output_name)


@hp.given(data=st.data())
def test_dependency_specs_input_name_semantics(data: st.DataObject) -> None:
    layer = data.draw(layers())
    input_name: str | EllipsisType | None = data.draw(
        st.one_of(st.none(), st.just(Ellipsis), names())
    )
    spec = data.draw(dependency_specs(layer, input_names=st.just(input_name)))
    if input_name is None:
        assert spec.resolved_input_name is None
    elif input_name is Ellipsis:
        assert spec.resolved_input_name == spec.output_name
    else:
        assert spec.resolved_input_name == input_name
