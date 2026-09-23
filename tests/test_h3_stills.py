import io
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
        ('UNETLoader', 'unet_name', 'minimax_h3_fl2va_pruned_int8_convrot.safetensors'),
        ('UNETLoader', 'unet_name', 'minimax_h3_ref2va_pruned_int8_convrot.safetensors'),
        ('CLIPLoader', 'clip_name', 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'),
        ('VAELoader', 'vae_name', 'minimax_h3_video_vae_fp16.safetensors'),
    ):
        info[loader]['input']['required'][field][0].append(filename)
    for name in ('MiniMaxH3ImageToVideo', 'MiniMaxH3ReferenceToVideo', 'ImageFromBatch', 'BasicScheduler'):
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
        self.assertEqual(nodes['UNETLoader']['unet_name'], 'minimax_h3_fl2va_pruned_int8_convrot.safetensors')
        inputs = nodes['MiniMaxH3ImageToVideo']
        self.assertEqual(inputs['length'], 5)
        self.assertEqual(inputs['width'], 672)
        self.assertEqual(inputs['height'], 384)
        self.assertNotIn('audio_vae', inputs)
        self.assertNotIn('first_frame', inputs)
        self.assertNotIn('last_frame', inputs)
        self.assertNotIn('MiniMaxH3ReferenceToVideo', nodes)
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
        self.assertEqual(nodes['UNETLoader']['unet_name'], 'minimax_h3_ref2va_pruned_int8_convrot.safetensors')
        self.assertEqual(inputs['ref_image_size'], 'match')
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

    def test_task_availability_requires_its_own_checkpoint_and_node(self):
        for task, other_task, model, node in (
            ('generation', 'editing', 'fl2va', 'MiniMaxH3ImageToVideo'),
            ('editing', 'generation', 'ref2va', 'MiniMaxH3ReferenceToVideo'),
        ):
            with self.subTest(task=task, missing='checkpoint'):
                info = h3_info()
                info['UNETLoader']['input']['required']['unet_name'][0].remove(
                    f'minimax_h3_{model}_pruned_int8_convrot.safetensors')
                self.assertFalse(model_profiles(info, task)['minimax_h3']['available'])
                self.assertTrue(model_profiles(info, other_task)['minimax_h3']['available'])
            with self.subTest(task=task, missing='node'):
                info = h3_info()
                del info[node]
                self.assertEqual(model_profiles(info, task)['minimax_h3']['missing'], [node])
                self.assertTrue(model_profiles(info, other_task)['minimax_h3']['available'])

    async def test_reference_detail_keeps_output_size_and_mask_independent(self):
        for detail, expected_sizes in (
            ('match', [[256, 128], [128, 256], [256, 128]]),
            ('max', [[1600, 800], [600, 1200], [1600, 800]]),
        ):
            with self.subTest(detail=detail), tempfile.TemporaryDirectory() as tmp:
                self.client.upload_image.reset_mock()
                canvas = self.sessions.create('default', 1600, 800, (10, 20, 30))
                reference = Path(tmp) / 'portrait.jpg'
                exif = Image.Exif()
                exif[274] = 6
                Image.new('RGB', (1200, 600), 'green').save(reference, exif=exif)
                result = await self.registry.tools['edit_image'](
                    'make the marked area blue', backend='minimax_h3', reference_paths=[str(reference)],
                    region=[400, 200, 400, 200], max_side=256, h3_reference_detail=detail)
                report = json.loads(result[0].text)
                self.assertEqual(report['generation_size'], [256, 128])
                self.assertEqual(report['canvas_size'], [1600, 800])
                self.assertEqual(report['reference_detail'], detail)
                self.assertEqual(report['reference_sizes'], expected_sizes)
                uploaded = [Image.open(io.BytesIO(call.args[0]))
                            for call in self.client.upload_image.await_args_list]
                self.assertEqual([list(image.size) for image in uploaded], expected_sizes)
                mask = uploaded[-1]
                self.assertEqual(mask.getpixel((0, 0))[:3], (0, 0, 0))
                self.assertEqual(mask.getpixel((mask.width * 3 // 8, mask.height * 3 // 8))[:3], (255, 255, 255))
                inputs = next(n['inputs'] for n in self.run.await_args.args[0].values()
                              if n['class_type'] == 'MiniMaxH3ReferenceToVideo')
                self.assertEqual((inputs['width'], inputs['height']), (256, 128))
                self.assertEqual(inputs['ref_image_size'], detail)
                self.assertEqual(canvas.composite().getpixel((600, 300)), (0, 0, 255, 255))
                self.assertEqual(canvas.composite().getpixel((0, 0)), (10, 20, 30, 255))

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
        detail = definitions['edit_image'].inputSchema['properties']['h3_reference_detail']
        self.assertEqual(detail['enum'], ['match', 'max'])
        self.assertEqual(detail['default'], 'match')
        with patch('prompt_rules.MASTER_PROMPT_DIR', Path(__file__).resolve().parents[1] / 'masters'):
            guide = get_prompt_guidance('minimax_h3', 'editing')
        self.assertEqual(len(guide['sources']), 1)
        self.assertEqual(Path(guide['sources'][0]['path']).name, 'minimax_h3_pseudo_image_master.txt')
        with self.assertRaisesRegex(ValueError, 'outpaint'):
            get_prompt_guidance('minimax_h3', 'outpaint')


if __name__ == '__main__':
    unittest.main()
