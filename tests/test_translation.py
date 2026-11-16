"""Tests for DagsterObjectTranslation."""

from dataclasses import FrozenInstanceError
from typing import Any

import dagster as dg
import hypothesis as hp
import hypothesis.strategies as st
import pytest as pt

from dagster_object_factory.layer import DependencySpec, Layer
from dagster_object_factory.translator import DagsterObjectTranslation

from .strategies import layers
from .toy_example import A, B, C


def _partitions_def() -> dg.StaticPartitionsDefinition:
    return dg.StaticPartitionsDefinition(["part1", "part2"])


def _translation(
    layer: Layer,
    specs: tuple[DependencySpec, ...] = (),
    output_names: tuple[str, ...] | None = None,
    name: str = "ab",
    description: str | None = None,
    code_version: str | None = None,
    group_name: str | None = None,
    partitions_def: dg.PartitionsDefinition | None = None,
    dagster_type: type = str,
    tags: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
    kinds: set[str] | None = None,
) -> DagsterObjectTranslation:
    if output_names is None:
        output_names = (min(layer.output_names),)
    return DagsterObjectTranslation(
        name=name,
        output_names=output_names,
        description=description,
        dependency_specs=specs,
        code_version=code_version,
        partitions_def=partitions_def,
        group_name=group_name,
        dagster_type=dagster_type,
        keys_by_output_name={
            output_name: layer.resolve_key(output_name) for output_name in output_names
        },
        descriptions_by_output_name={
            output_name: description for output_name in output_names
        },
        tags_by_output_name={output_name: tags for output_name in output_names},
        metadata_by_output_name={
            output_name: dict(metadata or {}) for output_name in output_names
        },
        kinds_by_output_name={output_name: kinds for output_name in output_names},
    )


@pt.fixture()
def translation(leaf_layer: Layer, composite_layer: Layer) -> DagsterObjectTranslation:
    specs = (
        DependencySpec(output_name=A.name, layer=leaf_layer),
        DependencySpec(output_name=C.name, layer=leaf_layer, input_name="renamed"),
        DependencySpec(output_name=B.name, layer=leaf_layer, input_name=None),
    )
    return _translation(
        composite_layer,
        specs=specs,
        output_names=("ab",),
        description="A composite",
        code_version="v1",
        group_name="composites",
        partitions_def=_partitions_def(),
        tags={"type": "composite"},
        metadata={"mixed": "yes"},
        kinds={"derived"},
    )


def test_immutable(translation: DagsterObjectTranslation) -> None:
    with pt.raises(FrozenInstanceError):
        translation.name = "other"  # type: ignore[reportAttributeAccessIssue]


def test_is_multi_output(composite_layer: Layer) -> None:
    single = _translation(composite_layer)
    multi = _translation(composite_layer, output_names=("ab", "bc"))
    assert not single.is_multi_output
    assert multi.is_multi_output


def test_keys_follow_output_order(composite_layer: Layer) -> None:
    output_names = ("bc", "ab", "ca")
    translation = _translation(composite_layer, output_names=output_names)
    assert translation.keys == [
        composite_layer.resolve_key(output_name) for output_name in output_names
    ]


def test_get_asset_spec(
    translation: DagsterObjectTranslation,
    composite_layer: Layer,
    leaf_layer: Layer,
) -> None:
    spec = translation.get_asset_spec("ab")
    assert spec.key == composite_layer.resolve_key("ab")
    assert spec.description == "A composite"
    assert spec.code_version == "v1"
    assert spec.group_name == "composites"
    assert spec.partitions_def is translation.partitions_def
    # AssetSpec derives dagster/kind tags from kinds.
    assert spec.tags == {"type": "composite", "dagster/kind/derived": ""}
    assert spec.metadata == {"mixed": "yes"}
    assert spec.kinds == {"derived"}
    # Deps carry every dependency spec, regardless of input name.
    assert {dep.asset_key for dep in spec.deps} == {
        leaf_layer.resolve_key(name) for name in ("a", "c", "b")
    }


def test_get_asset_spec_unknown_output_raises_key_error(
    translation: DagsterObjectTranslation,
) -> None:
    with pt.raises(KeyError):
        translation.get_asset_spec("not-there")


def test_get_asset_out(
    translation: DagsterObjectTranslation,
    composite_layer: Layer,
) -> None:
    out = translation.get_asset_out("ab", io_manager_key="custom")
    assert out.key == composite_layer.resolve_key("ab")
    assert out.io_manager_key == "custom"
    assert out.dagster_type is str


def test_get_asset_outs(
    translation: DagsterObjectTranslation,
    composite_layer: Layer,
) -> None:
    outs = translation.get_asset_outs(io_manager_key="custom")
    assert set(outs) == {"ab"}
    for output_name, out in outs.items():
        assert out.key == composite_layer.resolve_key(output_name)
        assert out.io_manager_key == "custom"


def test_get_asset_deps_preserves_order(
    translation: DagsterObjectTranslation,
    leaf_layer: Layer,
) -> None:
    deps = list(translation.get_asset_deps())
    assert [dep.asset_key for dep in deps] == [
        leaf_layer.resolve_key(name) for name in ("a", "c", "b")
    ]


def test_get_asset_deps_carries_metadata(
    leaf_layer: Layer, composite_layer: Layer
) -> None:
    spec = DependencySpec(
        output_name=A.name, layer=leaf_layer, metadata={"mixed": "yes"}
    )
    translation = _translation(composite_layer, specs=(spec,))
    (dep,) = list(translation.get_asset_deps())
    assert dep.metadata == {"mixed": "yes"}


def test_get_extra_deps(
    translation: DagsterObjectTranslation,
    leaf_layer: Layer,
) -> None:
    # Only the input-less spec (b) becomes a plain dep.
    assert [dep.asset_key for dep in translation.get_extra_deps()] == [
        leaf_layer.resolve_key(B.name)
    ]


def test_get_asset_ins(
    translation: DagsterObjectTranslation,
    leaf_layer: Layer,
) -> None:
    ins = translation.get_asset_ins()
    assert set(ins) == {"a", "renamed"}
    assert ins["a"].key == leaf_layer.resolve_key(A.name)
    assert ins["renamed"].key == leaf_layer.resolve_key(C.name)
    for asset_in in ins.values():
        dagster_type = asset_in.dagster_type
        assert isinstance(dagster_type, dg.DagsterType)
        assert dagster_type.typing_type is leaf_layer.python_type


def test_iter_output_values_single(translation: DagsterObjectTranslation) -> None:
    assert list(translation.iter_output_values("value")) == [("ab", "value")]


def test_iter_output_values_multi(composite_layer: Layer) -> None:
    translation = _translation(composite_layer, output_names=("ab", "bc"))
    assert list(translation.iter_output_values(("a", "b"))) == [
        ("ab", "a"),
        ("bc", "b"),
    ]


def test_iter_output_values_mismatch(composite_layer: Layer) -> None:
    translation = _translation(composite_layer, output_names=("ab", "bc", "ca"))
    with pt.raises(ValueError):
        list(translation.iter_output_values(("a", "b")))
    with pt.raises(ValueError):
        list(translation.iter_output_values(("a", "b", "c", "d")))


@hp.given(data=st.data())
def test_iter_output_values_aligns(data: st.DataObject) -> None:
    layer = data.draw(layers(min_outputs=1, max_outputs=5))
    output_names = tuple(data.draw(st.permutations(sorted(layer.output_names))))
    values = data.draw(
        st.lists(st.text(), min_size=len(output_names), max_size=len(output_names))
    )
    translation = _translation(layer, output_names=output_names)
    # A single-output result is the value itself; a multi-output result is a
    # sequence aligned with the output names.
    result: str | tuple[str, ...] = (
        values[0] if len(output_names) == 1 else tuple(values)
    )
    assert list(translation.iter_output_values(result)) == list(
        zip(output_names, values)
    )


@hp.given(data=st.data())
def test_iter_output_values_mismatch_raises(data: st.DataObject) -> None:
    layer = data.draw(layers(min_outputs=2, max_outputs=5))
    output_names = tuple(data.draw(st.permutations(sorted(layer.output_names))))
    values = data.draw(
        st.lists(st.text(), min_size=len(output_names), max_size=len(output_names))
    )
    translation = _translation(layer, output_names=output_names)
    for length in range(len(output_names)):
        with pt.raises(ValueError):
            list(translation.iter_output_values(tuple(values[:length])))
    with pt.raises(ValueError):
        list(translation.iter_output_values(tuple(values + ["extra"])))
