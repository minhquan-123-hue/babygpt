"""Minimal tests for Attention v3."""

import unittest

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


path = Path(__file__).with_name("12_attention_v3.py")
spec = spec_from_file_location("attention_v3", path)
module = module_from_spec(spec)
spec.loader.exec_module(module)


class AttentionV3Tests(unittest.TestCase):
    def test_each_token_gets_its_own_context_vector(self):
        embeddings = [
            [0.2, 0.8],
            [0.7, 0.1],
            [0.6, 0.3],
        ]

        results = module.attention(embeddings)

        # One output vector for each input token.
        self.assertEqual(len(results), len(embeddings))

        # Each output vector keeps the same dimensions.
        self.assertTrue(all(len(vector) == 2 for vector in results))

        # Each token has its own output list, not a shared list object.
        self.assertEqual(len({id(vector) for vector in results}), len(results))

        # With these example inputs, tokens receive different representations.
        self.assertNotEqual(results[0], results[1])
        self.assertNotEqual(results[1], results[2])


if __name__ == "__main__":
    unittest.main()
