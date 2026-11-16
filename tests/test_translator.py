"""Tests for DagsterObjectTranslator."""

from collections.abc import Mapping

import dagster as dg
import hypothesis as hp
import pytest as pt

from dagster_object_factory.layer import Layer, LayerDep
from dagster_object_factory.translator import DagsterObjectTranslator

from .conftest import (
    AB,
    CA,
    COMPOSITES,
    GROUP,
)
from .toy_example import Collection, Composite
from .toy_integration import CollectionTranslator, CompositeTranslator
from .worlds import World, worlds

LAYER = Layer(
    name="layer",
    key_prefix=(),
    python_type=str,
    output_names=frozenset({"output"}),
)


def test_abstract_translator_cannot_be_instantiated() -> None:
    with pt.raises(TypeError):
        DagsterObjectTranslator(layer=LAYER)  # type: ignore[abstract]


def test_missing_hook_cannot_be_instantiated(composite_layer: Layer) -> None:
    class _Incomplete(DagsterObjectTranslator[Composite, str]):
        def get_name(self, obj: Composite, /) -> str:
            return obj.name

    with pt.raises(TypeError):
        _Incomplete(composite_layer)  # type: ignore[abstract]


def test_call_wires_up_all_hooks(composite_layer: Layer, leaf_layer: Layer) -> None:
    translation = CompositeTranslator(composite_layer)(AB)
    assert translation.name == "ab"
    assert translation.output_names == ("ab",)
    assert translation.description is None
    assert translation.code_version is None
    assert translation.group_name == "composite"
    assert translation.dagster_type is composite_layer.python_type
    assert translation.partitions_def is composite_layer.partitions_def
    assert translation.keys == [composite_layer.resolve_key("ab")]
    assert {spec.key for spec in translation.dependency_specs} == {
        leaf_layer.resolve_key("a"),
        leaf_layer.resolve_key("b"),
    }


def test_partitioned_layer_flows_into_translation(leaf_layer: Layer) -> None:
    partitions_def = dg.TimeWindowPartitionsDefinition(
        start="2024-01-01",
        fmt="%Y-%m-%d",
        end="2024-01-05",
        cron_schedule="0 0 * * *",
        timezone="UTC",
    )
    layer = Layer(
        name="composite",
        key_prefix=("item",),
        python_type=str,
        output_names=frozenset(composite.name for composite in COMPOSITES),
        layer_deps=(LayerDep(layer=leaf_layer),),
        partitions_def=partitions_def,
    )
    translation = CompositeTranslator(layer)(AB)
    assert translation.partitions_def is partitions_def


def test_hook_overrides_reflected_in_translation(composite_layer: Layer) -> None:
    class _Overriding(CompositeTranslator):
        def get_description(self, obj: Composite, /) -> str | None:
            return "custom description"

        def get_code_version(self, obj: Composite, /) -> str | None:
            return "v42"

        def get_group_name(self, obj: Composite, /) -> str | None:
            return "custom group"

        def get_tags(self, obj: Composite, /) -> dict[str, str]:
            return {"custom": "tag"}

        def get_kinds(self, obj: Composite, /) -> set[str]:
            return {"custom"}

        def get_metadata(self, obj: Composite, /) -> Mapping[str, dg.MetadataValue]:
            return {"custom": dg.MetadataValue.text("value")}

    translation = _Overriding(composite_layer)(AB)
    assert translation.description == "custom description"
    assert translation.code_version == "v42"
    assert translation.group_name == "custom group"
    assert translation.tags_by_output_name["ab"] == {"custom": "tag"}
    assert translation.kinds_by_output_name["ab"] == {"custom"}
    assert translation.metadata_by_output_name["ab"] == {
        "custom": dg.MetadataValue.text("value")
    }


def test_by_output_name_defaults_broadcast(collection_layer: Layer) -> None:
    class _Broadcasting(CollectionTranslator):
        def get_description(self, obj: Collection, /) -> str | None:
            return "shared description"

        def get_tags(self, obj: Collection, /) -> dict[str, str]:
            return {"shared": "tag"}

        def get_kinds(self, obj: Collection, /) -> set[str]:
            return {"shared"}

        def get_metadata(self, obj: Collection, /) -> Mapping[str, dg.MetadataValue]:
            return {"shared": dg.MetadataValue.text("value")}

    translation = _Broadcasting(collection_layer)(GROUP)
    output_names = translation.output_names
    assert output_names == ("a", "b", "c")
    assert translation.descriptions_by_output_name == {
        name: "shared description" for name in output_names
    }
    assert translation.tags_by_output_name == {
        name: {"shared": "tag"} for name in output_names
    }
    assert translation.kinds_by_output_name == {
        name: {"shared"} for name in output_names
    }
    assert translation.metadata_by_output_name == {
        name: {"shared": dg.MetadataValue.text("value")} for name in output_names
    }


def test_default_hooks_broadcast_defaults(collection_layer: Layer) -> None:
    translation = CollectionTranslator(collection_layer)(GROUP)
    output_names = translation.output_names
    assert translation.descriptions_by_output_name == {
        name: None for name in output_names
    }
    assert translation.tags_by_output_name == {name: {} for name in output_names}
    assert translation.kinds_by_output_name == {name: None for name in output_names}
    assert translation.metadata_by_output_name == {name: {} for name in output_names}


def test_undeclared_output_raises(composite_layer: Layer) -> None:
    class _BadOutput(CompositeTranslator):
        def get_output_names(self, obj: Composite, /) -> tuple[str, ...]:
            return ("not-there",)

    with pt.raises(ValueError, match="not a declared output"):
        _BadOutput(composite_layer)(AB)


def test_group_name_defaults_to_layer_name(composite_layer: Layer) -> None:
    translation = CompositeTranslator(composite_layer)(CA)
    assert translation.group_name == composite_layer.name


@hp.given(world=worlds())
def test_world_translation_invariants(world: World) -> None:
    translator = CompositeTranslator(world.composite_layer)
    for composite in world.composites:
        translation = translator(composite)
        assert translation.name == composite.name
        assert translation.output_names == (composite.name,)
        assert translation.keys == [world.composite_layer.resolve_key(composite.name)]
        assert {spec.key for spec in translation.dependency_specs} == {
            world.leaf_layer.resolve_key(part.name) for part in composite.parts
        }
        assert translation.group_name == world.composite_layer.name
        assert translation.dagster_type is world.composite_layer.python_type
        assert translation.partitions_def is world.composite_layer.partitions_def


@hp.given(world=worlds().filter(lambda w: w.collections))
def test_world_collection_translation_invariants(world: World) -> None:
    translator = CollectionTranslator(world.collection_layer)
    for collection in world.collections:
        translation = translator(collection)
        expected_outputs = tuple(member.name for member in collection.members)
        assert translation.output_names == expected_outputs
        assert translation.keys == [
            world.collection_layer.resolve_key(name) for name in expected_outputs
        ]
        assert translation.is_multi_output == (len(collection.members) > 1)
        assert {spec.key for spec in translation.dependency_specs} == {
            world.leaf_layer.resolve_key(member.name) for member in collection.members
        }
