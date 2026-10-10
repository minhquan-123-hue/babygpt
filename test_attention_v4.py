"""Minimal tests for Attention v4's learnable scoring.""" 

import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


path = Path(__file__).with_name("12_attention_v4.py")
spec = spec_from_file_location("attention_v4", path)
module = module_from_spec(spec)
spec.loader.exec_module(module)


class AttentionV4Tests(unittest.TestCase):
    def test_each_token_gets_a_context_representation(self):
        embeddings = [
            [0.2, 0.8],
            [0.7, 0.1],
            [0.6, 0.3],
        ]

        results = module.attention(embeddings, [1.0, 1.0])

        self.assertEqual(len(results), len(embeddings))
        self.assertTrue(
            all(len(result["representation"]) == 2 for result in results)
        )

    def test_changing_parameters_changes_weights_and_context(self):
        embeddings = [
            [0.2, 0.8],
            [0.7, 0.1],
            [0.6, 0.3],
        ]

        initial = module.attention(embeddings, [1.0, 1.0])
        changed = module.attention(embeddings, [10.0, 0.1])

        # A parameter change affects attention weights and output vectors.
        self.assertNotEqual(initial[0]["weights"], changed[0]["weights"])
        self.assertNotEqual(
            initial[0]["representation"],
            changed[0]["representation"],
        )


if __name__ == "__main__":
    unittest.main()
