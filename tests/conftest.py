import logging

import dagster as dg
import pytest as pt

from dagster_object_factory.layer import Layer, LayerDep

from .toy_example import AB, BC, CA, GROUP, A, B, C

LEAVES = A, B, C
COMPOSITES = AB, BC, CA

PYTHON_TYPE = str
KEY_PREFIX = ("item",)
COLLECTION_KEY_PREFIX = ("group",)


@pt.fixture(autouse=True)
def _quiet_dagster_logs():
    logger = logging.getLogger("dagster")
    original_level = logger.level
    logger.setLevel(logging.ERROR)
    yield
    logger.level = original_level


@pt.fixture()
def leaf_layer() -> Layer:
    return Layer(
        name="leaf",
        description="Leaves",
        python_type=PYTHON_TYPE,
        output_names=frozenset(leaf.name for leaf in LEAVES),
        key_prefix=KEY_PREFIX,
    )


@pt.fixture()
def composite_layer(leaf_layer) -> Layer:
    return Layer(
        name="composite",
        description="Composites of exactly 2 leaves.",
        python_type=PYTHON_TYPE,
        output_names=frozenset(composite.name for composite in COMPOSITES),
        key_prefix=KEY_PREFIX,
        layer_deps=(LayerDep(layer=leaf_layer),),
    )


@pt.fixture()
def collection_layer(leaf_layer) -> Layer:
    return Layer(
        name="collection",
        description="Collection of leaves.",
        key_prefix=COLLECTION_KEY_PREFIX,
        python_type=PYTHON_TYPE,
        output_names=frozenset(member.name for member in GROUP.members),
        layer_deps=(LayerDep(layer=leaf_layer),),
    )


@pt.fixture()
def load_context() -> dg.ComponentLoadContext:
    return dg.ComponentTree.for_test().load_context


@pt.fixture()
def instance():
    with dg.instance_for_test() as test_instance:
        yield test_instance
