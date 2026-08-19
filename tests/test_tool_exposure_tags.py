import unittest

from Quasar.ai_tools.registry import ToolCategory, ToolRegistry


class _DummyTool:
    name = "dummy.public"
    description = "dummy"


class ToolExposureTagTests(unittest.TestCase):
    def test_loader_tags_are_preserved_on_registered_tools(self):
        registry = ToolRegistry()
        registry.register_loader(
            lambda: [_DummyTool()],
            category=ToolCategory.OTHER,
            requires_config=False,
            tags={"public", "conversation"},
            source="tests.exposure",
        )

        spec = registry._loaders[0]
        registry.register(
            _DummyTool(),
            category=spec.category,
            dependencies=spec.dependencies,
            source=spec.source,
            tags=spec.tags,
        )

        metadata = registry.get_metadata("dummy.public")
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.tags, {"public", "conversation"})
        self.assertEqual([tool.name for tool in registry.get_by_tag("public")], ["dummy.public"])


if __name__ == "__main__":
    unittest.main()
