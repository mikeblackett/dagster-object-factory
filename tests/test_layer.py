from dataclasses import FrozenInstanceError

import dagster as dg
import hypothesis as hp
import pytest as pt

from dagster_object_factory.layer import Layer, LayerDep

from .strategies import asset_keys


def test_immutability(leaf_layer: Layer) -> None:
    with pt.raises(FrozenInstanceError):
        leaf_layer.name = "value"  # type: ignore[reportAttributeAccessIssue]
    assert isinstance(leaf_layer.output_names, frozenset)


def test_contains_reflects_output_names(leaf_layer: Layer) -> None:
    for name in leaf_layer.output_names:
        assert name in leaf_layer
    assert "outsider" not in leaf_layer


## resolve_key
@hp.given(key=asset_keys())
def test_key_resolution(key: dg.AssetKey) -> None:
    output_name = key.path[-1]
    layer = Layer(
        name="layer",
        key_prefix=tuple(key.path[:-1]),
        python_type=str,
        output_names=frozenset({output_name}),
    )
    assert layer.resolve_key(output_name) == key


def test_key_resolution_raises_value_error(leaf_layer: Layer) -> None:
    with pt.raises(ValueError, match="not a declared output"):
        leaf_layer.resolve_key("not-there")


## resolve_layer_dep
def test_ambiguous_layer_dependency_raises_value_error(leaf_layer: Layer) -> None:
    dupe_layer = Layer(
        name="dupe",
        key_prefix=(),
        python_type=str,
        output_names=frozenset(["duped"]),
        layer_deps=(LayerDep(layer=leaf_layer), LayerDep(layer=leaf_layer)),
    )
    output_name = next(iter(leaf_layer.output_names))
    with pt.raises(ValueError, match="ambiguous"):
        dupe_layer.resolve_layer_dep(output_name)


def test_failed_layer_dependency_resolution_returns_none(leaf_layer: Layer) -> None:
    assert leaf_layer.resolve_layer_dep("not-there") is None


def test_layer_dependency_resolution(leaf_layer: Layer, composite_layer: Layer) -> None:
    output_name = next(iter(leaf_layer.output_names))
    layer_dep = composite_layer.resolve_layer_dep(output_name)
    assert layer_dep is not None
    assert layer_dep.layer is leaf_layer
