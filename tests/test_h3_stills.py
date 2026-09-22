import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

import server
from editing import model_profiles, png_bytes, register_editing_tools
from prompt_rules import get_prompt_guidance
from session import SessionManager
from test_editing import Registry, node_info


def h3_info():
    info = node_info()
    for loader, field, filename in (
        ('UNETLoader', 'unet_name', 'minimax_h3_ref2va_pruned_int8_convrot.safetensors'),
        ('CLIPLoader', 'clip_name', 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'),
        ('VAELoader', 'vae_name', 'minimax_h3_video_vae_fp16.safetensors'),
    ):
        info[loader]['input']['required'][field][0].append(filename)
    for name in ('MiniMaxH3ReferenceToVideo', 'ImageFromBatch', 'BasicScheduler'):
        info[name] = {}
    return info


class H3StillTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = AsyncMock()
        self.client.get_object_info.return_value = h3_info()
        self.client.upload_image.side_effect = lambda data, name: name
        self.sessions = SessionManager()
        self.image = png_bytes(Image.new('RGB', (64, 64), 'blue'))
        self.run = AsyncMock(return_value=self.image)
        self.registry = Registry()
        register_editing_tools(self.registry, self.client, self.sessions, self.run)
        for name, value in (('comfy', self.client), ('sessions', self.sessions), ('run_workflow', self.run)):
            p = patch.object(server, name, value)
            p.start()
            self.addCleanup(p.stop)

    async def test_generation_saves_one_frame_without_optional_dependencies(self):
        result = await server.generate_image('one mug', model='minimax_h3', width=641, height=383,
                                             seed=0, cfg=7, negative_prompt='watermark')
        self.assertIn('Model: minimax_h3', result[0].text)
        graph = self.run.await_args.args[0]
        nodes = {n['class_type']: n['inputs'] for n in graph.values()}
        self.assertEqual(nodes['MiniMaxH3ReferenceToVideo']['length'], 5)
        self.assertEqual(nodes['MiniMaxH3ReferenceToVideo']['width'], 672)
        self.assertEqual(nodes['MiniMaxH3ReferenceToVideo']['height'], 384)
        self.assertNotIn('audio_vae', nodes['MiniMaxH3ReferenceToVideo'])
        self.assertNotIn('LoadImage', nodes)
        self.assertNotIn('VAEDecodeAudio', nodes)
        self.assertEqual(nodes['RandomNoise']['noise_seed'], 0)
        self.assertEqual(nodes['BasicScheduler']['steps'], 20)
        saved = graph[nodes['SaveImage']['images'][0]]
        self.assertEqual(saved['class_type'], 'ImageFromBatch')
        self.assertEqual(saved['inputs']['batch_index'], 0)
        self.assertEqual(saved['inputs']['length'], 1)
        self.assertEqual(self.run.await_args.kwargs['timeout'], 3600)
        self.assertIn('negative_prompt', result[1].text)
        self.assertIn('supplied cfg is unused', result[1].text)

    async def test_reference_order_mask_pixels_and_undo(self):
        canvas = self.sessions.create('default', 64, 64, (10, 20, 30))
        original = canvas.composite().tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / 'reference.png'
            Image.new('RGB', (64, 64), 'green').save(reference)
            result = await self.registry.tools['edit_image'](
                'use the color from <Picture 2>', backend='minimax_h3', reference_paths=[str(reference)],
                region=[16, 16, 32, 32], seed=0, steps=23, cfg=5, timeout=123)
        report = json.loads(result[0].text)
        nodes = {n['class_type']: n['inputs'] for n in self.run.await_args.args[0].values()}
        inputs = nodes['MiniMaxH3ReferenceToVideo']
        self.assertIn('Edit <Picture 1>', inputs['prompt'])
        self.assertIn('<Picture 3> is an edit mask', inputs['prompt'])
        self.assertEqual(len([k for k in inputs if k.startswith('ref_images.')]), 3)
        self.assertEqual(nodes['BasicScheduler']['steps'], 23)
        self.assertEqual(self.run.await_args.kwargs['timeout'], 123)
        self.assertEqual(report['cfg'], 1.0)
        self.assertIn('supplied cfg is unused', report['sampling_note'])
        self.assertEqual(canvas.composite().getpixel((32, 32)), (0, 0, 255, 255))
        for y in range(64):
            for x in range(64):
                if not (16 <= x < 48 and 16 <= y < 48):
                    self.assertEqual(canvas.composite().getpixel((x, y)), (10, 20, 30, 255))
        canvas.undo()
        self.assertEqual(canvas.composite().tobytes(), original)

    async def test_missing_h3_does_not_submit_or_substitute(self):
        info = h3_info()
        del info['ImageFromBatch']
        self.client.get_object_info.return_value = info
        result = await server.generate_image('mug', model='minimax_h3')
        self.assertIn('minimax_h3 is unavailable', result[0].text)
        self.run.assert_not_awaited()
        self.assertNotIn('minimax_h3', model_profiles(h3_info(), 'outpaint'))

    async def test_h3_batch_exports_and_reports_effective_settings(self):
        self.client.batch_run_workflows.return_value = [
            {'history': {'_cached_file_bytes': self.image}, 'error': None}]
        with tempfile.TemporaryDirectory() as tmp:
            result = await server.batch_generate_tool(
                [{'model': 'minimax_h3', 'prompt': 'mug', 'cfg': 7, 'seed': 0}], export_dir=tmp)
            record = json.loads(result[0].text)['results'][0]
            self.assertEqual(record['status'], 'ok')
            self.assertEqual(record['cfg'], 1.0)
            self.assertEqual(record['steps'], 20)
            self.assertTrue(Path(record['file']).is_file())
        self.assertEqual(self.client.batch_run_workflows.await_args.kwargs['timeout'], 3600)

    async def test_background_jobs_and_tool_schemas_accept_h3(self):
        with patch.object(server.generation_jobs, 'submit', return_value={'job_id': 'test'}) as submit:
            await server.submit_generation_job([{'prompt': 'mug', 'model': 'minimax_h3'}])
            submit.assert_called_once()
        definitions = {tool.name: tool for tool in await server.app.list_tools()}
        for name, arg in (('generate_image', 'model'), ('edit_image', 'backend'), ('get_prompt_guidance', 'model')):
            self.assertIn('minimax_h3', definitions[name].inputSchema['properties'][arg]['enum'])
        self.assertIn('<Picture 1>', definitions['edit_image'].description)
        guide = get_prompt_guidance('minimax_h3', 'editing')
        self.assertEqual(guide['sources'], [])
        with self.assertRaisesRegex(ValueError, 'outpaint'):
            get_prompt_guidance('minimax_h3', 'outpaint')


if __name__ == '__main__':
    unittest.main()
