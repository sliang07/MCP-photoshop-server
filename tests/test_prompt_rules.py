import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

import prompt_rules
import server
from editing import png_bytes, register_editing_tools
from session import SessionManager
from test_editing import node_info, Registry


class PromptRulesTests(unittest.IsolatedAsyncioTestCase):
    def test_aesthetic_removes_quality_scores_but_keeps_exact_lettering(self):
        prompt = 'score_7, best quality, 1girl. Her sign reads "score_7, score_2, HELLO!" in soft studio lighting.'
        negative = "worst quality, score_1, score_2, jpeg artifacts"
        positive, negative, changes = prompt_rules.prepare_prompts(
            "anima", "anima-aesthetic-v1.1.safetensors", prompt, negative)
        self.assertEqual(positive, 'best quality, 1girl. Her sign reads "score_7, score_2, HELLO!" in soft studio lighting.')
        self.assertEqual(negative, "worst quality, jpeg artifacts")
        self.assertEqual(len(changes), 1)

    def test_quotes_apostrophes_and_case_are_preserved(self):
        for text in ('"score_7, score_3"', "'score_7, score_3'", "“score_7, score_3”", "‘score_7, score_3’", "`score_7, score_3`"):
            original = f"score_7, a girl's sign reads {text}, keep HELLO!"
            result, _, _ = prompt_rules.prepare_prompts("anima", "anima-aesthetic-v1.1.safetensors", original, "")
            self.assertEqual(result, f"a girl's sign reads {text}, keep HELLO!")

    def test_base_scores_and_other_models_are_not_rewritten(self):
        for model, checkpoint in (("anima", "folder/anima-base-v1.0.safetensors"),
                                  ("qwen21", "qwen.safetensors")):
            original = "score_7, blue coat, soft studio lighting"
            positive, negative, changes = prompt_rules.prepare_prompts(model, checkpoint, original, "score_1")
            self.assertEqual((positive, negative, changes), (original, "score_1", []))

    def test_flux_dev_reports_unused_negative_without_rewriting_positive(self):
        positive = 'A sign reads "score_7" beside a blue bird.'
        actual, negative, changes = prompt_rules.prepare_prompts("flux2", "flux2-dev-nvfp4.safetensors", positive, "watermark")
        self.assertEqual(actual, positive)
        self.assertEqual(negative, "")
        self.assertIn("does not use negative_prompt", changes[0])

    def test_only_single_outer_code_fences_are_removed(self):
        self.assertEqual(prompt_rules.unwrap_prompt('```text\nA sign reads "HELLO!"\n```'), 'A sign reads "HELLO!"')
        for literal in ('A sign displays ```code```.', '```first\ntext\n```',
                        '```\npositive\n```\n```\nnegative\n```'):
            self.assertEqual(prompt_rules.unwrap_prompt(literal), literal)

    def test_descriptive_prose_and_long_prompts_are_not_changed(self):
        prompt = "The first person wears a blue coat; the second wears red. Intentionally blurred, soft studio lighting. " * 30
        for model in ("flux2", "qwen21", "anima"):
            actual, negative, changes = prompt_rules.prepare_prompts(model, "anima-aesthetic-v1.1.safetensors", prompt, "")
            self.assertEqual((actual, negative, changes), (prompt, "", []))

    def test_full_master_is_read_fresh_and_paths_are_fixed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(prompt_rules, "MASTER_PROMPT_DIR", tmp):
            path = Path(tmp) / "flux2prompt.txt"
            path.write_text("Current master: preserve studio lighting.", encoding="utf-8")
            first = prompt_rules.get_prompt_guidance("flux2", "generation")
            self.assertEqual(first["sources"][0]["text"], path.read_text())
            path.write_text("Revised master: keep exact lettering.", encoding="utf-8")
            second = prompt_rules.get_prompt_guidance("flux2", "generation")
            self.assertNotEqual(first["sources"][0]["sha256"], second["sources"][0]["sha256"])
            with self.assertRaisesRegex(ValueError, "Choose model"):
                prompt_rules.get_prompt_guidance("../secret", "generation")
            with self.assertRaisesRegex(ValueError, "Cannot read master prompt"):
                prompt_rules.get_prompt_guidance("anima", "generation")

    def test_anima_edit_is_not_advertised_by_guide(self):
        with self.assertRaisesRegex(ValueError, "generation only"):
            prompt_rules.get_prompt_guidance("anima", "editing")

    def test_qwen_selects_full_master_by_task_and_reads_updates(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(prompt_rules, "MASTER_PROMPT_DIR", tmp):
            t2i = Path(tmp) / "qwen_image_2.1_system_prompt_t2i.txt"
            edit = Path(tmp) / "qwen_image_2.1_system_prompt_edit.txt"
            t2i.write_text('Generation: retain "夏日特惠".', encoding="utf-8-sig")
            edit.write_text('Editing: retain "ลดราคา".', encoding="utf-8")
            for task, expected in (("generation", t2i), ("editing", edit), ("outpaint", edit)):
                result = prompt_rules.get_prompt_guidance("qwen21", task)
                self.assertEqual(len(result["sources"]), 1)
                source = result["sources"][0]
                self.assertEqual(Path(source["path"]), expected)
                self.assertEqual(source["text"], expected.read_text(encoding="utf-8-sig"))
            old_hash = result["sources"][0]["sha256"]
            edit.write_text("Updated edit instructions.", encoding="utf-8")
            updated = prompt_rules.get_prompt_guidance("qwen21", "editing")
            self.assertEqual(updated["sources"][0]["text"], "Updated edit instructions.")
            self.assertNotEqual(updated["sources"][0]["sha256"], old_hash)

    def test_missing_qwen_master_does_not_fall_back_to_other_models(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(prompt_rules, "MASTER_PROMPT_DIR", tmp):
            for filename in ("flux2prompt.txt", "anima_prompt.txt", "qwen_image_2.1_system_prompt_t2i.txt"):
                (Path(tmp) / filename).write_text("Other guide.", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "qwen_image_2.1_system_prompt_edit.txt"):
                prompt_rules.get_prompt_guidance("qwen21", "editing")

    def test_h3_master_is_read_fresh_for_both_tasks(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(prompt_rules, "MASTER_PROMPT_DIR", tmp):
            path = Path(tmp) / "minimax_h3_pseudo_image_master.txt"
            path.write_text('H3 stills: keep "4K" as literal lettering.', encoding="utf-8-sig")
            for task in ("generation", "editing"):
                result = prompt_rules.get_prompt_guidance("minimax_h3", task)
                self.assertEqual(result["sources"][0]["text"], path.read_text(encoding="utf-8-sig"))
                self.assertEqual(Path(result["sources"][0]["path"]), path)
            old_hash = result["sources"][0]["sha256"]
            path.write_text('Revised H3 master.', encoding="utf-8")
            self.assertNotEqual(prompt_rules.get_prompt_guidance("minimax_h3", "editing")["sources"][0]["sha256"], old_hash)
            path.unlink()
            for task in ("generation", "editing"):
                with self.assertRaisesRegex(ValueError, "Cannot read master prompt.*minimax_h3_pseudo_image_master"):
                    prompt_rules.get_prompt_guidance("minimax_h3", task)
            with self.assertRaisesRegex(ValueError, "H3 supports generation and editing"):
                prompt_rules.get_prompt_guidance("minimax_h3", "outpaint")

    async def test_h3_master_rules_reach_generation_and_edit_tools(self):
        definitions = {tool.name: tool for tool in await server.app.list_tools()}
        for name in ("generate_image", "batch_generate", "edit_image"):
            description = definitions[name].description
            self.assertIn("minimax_h3_pseudo_image_master.txt", description)
            self.assertIn("decoded rewritten_prompt as prompt", description)
            self.assertIn("do not invent factual data", description)
        for name in ("generate_image", "batch_generate"):
            self.assertIn("H3 generation: open with medium", definitions[name].description)
            self.assertNotIn("H3 editing: lead with", definitions[name].description)
        self.assertIn("H3 editing: lead with", definitions['edit_image'].description)
        self.assertNotIn("H3 master (", definitions['outpaint'].description)

    async def test_rules_are_in_tool_descriptions_without_initialize(self):
        definitions = {tool.name: tool for tool in await server.app.list_tools()}
        for name in ("generate_image", "edit_image", "outpaint", "batch_generate"):
            description = definitions[name].description
            self.assertIn("MASTER PROMPT RULES", description)
            self.assertIn("exact visible lettering", description)
            self.assertIn("get_prompt_guidance", description)
            self.assertIn("not a prerequisite", description)
        for name in ("generate_image", "batch_generate"):
            self.assertIn("omit score_* quality tags from BOTH", definitions[name].description)
            self.assertIn("30–80 words", definitions[name].description)
            self.assertIn("qwen_image_2.1_system_prompt_t2i.txt", definitions[name].description)
            self.assertIn("400–500 words", definitions[name].description)
            self.assertNotIn("qwen_image_2.1_system_prompt_edit.txt", definitions[name].description)
        for name in ("edit_image", "outpaint"):
            self.assertIn("qwen_image_2.1_system_prompt_edit.txt", definitions[name].description)
            self.assertIn("without tags", definitions[name].description)
            self.assertIn("input's dominant text language", definitions[name].description)
            self.assertNotIn("qwen_image_2.1_system_prompt_t2i.txt", definitions[name].description)

    async def test_generation_submits_cleaned_separate_prompts_and_reports_changes(self):
        client = AsyncMock()
        client.get_object_info.return_value = node_info()
        run = AsyncMock(return_value=png_bytes(Image.new("RGBA", (64, 64), "red")))
        sessions = SessionManager()
        with patch.object(server, "comfy", client), patch.object(server, "sessions", sessions), \
             patch.object(server, "run_workflow", run):
            result = await server.generate_image('```\nscore_7, a blue bird\n```', model="anima",
                                                  negative_prompt="score_1, jpeg artifacts")
        encoders = [n["inputs"]["text"] for n in run.await_args.args[0].values() if n["class_type"] == "CLIPTextEncode"]
        self.assertEqual(encoders, ["a blue bird", "jpeg artifacts"])
        report = json.loads(result[1].text)
        self.assertEqual(report["effective_prompt"], "a blue bird")
        self.assertEqual(len(report["prompt_adjustments"]), 2)

    async def test_mixed_batch_applies_rules_per_checkpoint(self):
        client = AsyncMock()
        client.get_object_info.return_value = node_info()
        client.batch_run_workflows.return_value = [{"history": None, "error": "test"}] * 2
        with tempfile.TemporaryDirectory() as tmp, patch.object(server, "comfy", client):
            result = await server.batch_generate_tool(
                [{"prompt": "score_7, a bird", "model": model, "negative_prompt": "score_1, artifacts"}
                 for model in ("anima", "flux2")], export_dir=tmp)
        graphs = client.batch_run_workflows.await_args.args[0]
        texts = [[n["inputs"]["text"] for n in graph.values() if n["class_type"] == "CLIPTextEncode"] for graph in graphs]
        # flux2 (dev) is guidance-distilled: no negative conditioning is encoded.
        self.assertEqual(texts, [["a bird", "artifacts"], ["score_7, a bird"]])
        records = json.loads(result[0].text)["results"]
        self.assertIn("prompt_adjustments", records[0])
        self.assertIn("does not use negative_prompt", records[1]["prompt_adjustments"][0])
        self.assertEqual(records[1]["negative_prompt"], "")

    async def test_editor_unwraps_prompt_without_changing_exact_text(self):
        registry = Registry()
        sessions = SessionManager()
        sessions.create("default", 64, 64)
        client = AsyncMock()
        client.get_object_info.return_value = node_info()
        client.upload_image.return_value = "source.png"
        run = AsyncMock(return_value=png_bytes(Image.new("RGBA", (64, 64), "red")))
        register_editing_tools(registry, client, sessions, run)
        result = await registry.tools["edit_image"]('```\nChange the sign to "HELLO, score_7!".\n```')
        effective = json.loads(result[0].text)["effective_prompt"]
        self.assertNotIn("```", effective)
        self.assertIn('Change the sign to "HELLO, score_7!".', effective)
        self.assertTrue(effective.startswith("Edit the image."))
        self.assertNotIn("<image1>", effective)
        self.assertNotIn("\n", effective)


if __name__ == "__main__":
    unittest.main()
