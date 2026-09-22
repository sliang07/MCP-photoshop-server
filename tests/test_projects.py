import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

from canvas import Canvas
from project import load_project, save_project
import server
from session import SessionManager


def snapshot(canvas):
    return (canvas.width, canvas.height, canvas.active_layer_index,
            [layer.to_dict() for layer in canvas.layers])


def sample_canvas():
    canvas = Canvas(4, 3, (10, 20, 30, 90))
    canvas.add_layer("Subject \u732b", Image.new("RGBA", (4, 3), (80, 90, 100, 140)), 0.45, "Multiply")
    canvas.layers[1].image.putpixel((1, 1), (20, 30, 40, 0))
    canvas.layers[1].mask = Image.new("L", (4, 3), 170)
    canvas.layers[1].mask.putpixel((1, 1), 0)
    canvas.layers[1].visible = False
    canvas.active_layer_index = 0
    canvas._save_state()
    return canvas


def write_archive(path, manifest, members):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest).encode("utf-8"))
        for name, data in members.items():
            archive.writestr(name, data)


class ProjectTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "document.mcpproj"
        self.canvas = sample_canvas()
        save_project(self.canvas, self.path)
        with zipfile.ZipFile(self.path) as archive:
            self.manifest = json.loads(archive.read("manifest.json"))
            self.members = {name: archive.read(name) for name in archive.namelist() if name != "manifest.json"}

    def test_roundtrip_preserves_document_and_starts_fresh_history(self):
        loaded = load_project(self.path)
        self.assertEqual(snapshot(loaded), snapshot(self.canvas))
        self.assertEqual(loaded.composite().tobytes(), self.canvas.composite().tobytes())
        self.assertEqual(len(loaded.undo_stack), 1)
        self.assertFalse(loaded.undo())
        loaded.layers[1].mask.putpixel((1, 1), 255)
        self.assertEqual(self.canvas.layers[1].mask.getpixel((1, 1)), 0)

    def test_failed_save_preserves_previous_file_and_removes_temporary(self):
        original = self.path.read_bytes()
        for target in ("project.os.replace", "project.os.fsync", "canvas.Layer.to_dict"):
            with self.subTest(target=target), patch(target, side_effect=OSError("simulated failure")):
                with self.assertRaises(OSError):
                    save_project(self.canvas, self.path)
            self.assertEqual(self.path.read_bytes(), original)
            self.assertEqual(set(self.directory.iterdir()), {self.path})

    def test_unsafe_member_references_are_rejected(self):
        for value in ("../outside.png", "/absolute.png", "C:/outside.png", "layers\\0.png"):
            for field in ("image", "mask"):
                with self.subTest(path=value, field=field):
                    manifest = copy.deepcopy(self.manifest)
                    manifest["layers"][0][field] = value
                    write_archive(self.path, manifest, self.members)
                    with self.assertRaisesRegex(ValueError, "path"):
                        load_project(self.path)
        self.assertEqual(set(self.directory.iterdir()), {self.path})

    def test_bad_manifest_values_are_rejected(self):
        cases = [("version", 2), ("version", True), ("width", 0), ("height", True),
                 ("active_layer", -1), ("active_layer", 2), ("layers", [])]
        for key, value in cases:
            with self.subTest(field=key, value=value):
                manifest = copy.deepcopy(self.manifest)
                manifest[key] = value
                write_archive(self.path, manifest, self.members)
                with self.assertRaises(ValueError):
                    load_project(self.path)
        for key, value in [("visible", 1), ("opacity", float("nan")), ("name", 123), ("blend_mode", "unsupported")]:
            with self.subTest(layer_field=key):
                manifest = copy.deepcopy(self.manifest)
                manifest["layers"][0][key] = value
                write_archive(self.path, manifest, self.members)
                with self.assertRaises(ValueError):
                    load_project(self.path)
        del self.manifest["layers"][0]["mask"]
        write_archive(self.path, self.manifest, self.members)
        with self.assertRaisesRegex(ValueError, "mask"):
            load_project(self.path)

    def test_missing_duplicate_and_invalid_png_members_are_rejected(self):
        members = dict(self.members)
        del members["layers/1.png"]
        write_archive(self.path, self.manifest, members)
        with self.assertRaisesRegex(ValueError, "missing member"):
            load_project(self.path)
        write_archive(self.path, self.manifest, self.members)
        with zipfile.ZipFile(self.path, "a") as archive, self.assertWarns(UserWarning):
            archive.writestr("layers/0.png", self.members["layers/0.png"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_project(self.path)
        for member, image in [("layers/0.png", Image.new("RGB", (4, 3))),
                              ("layers/1.png", Image.new("RGBA", (2, 2))),
                              ("masks/1.png", Image.new("RGBA", (4, 3)))]:
            with self.subTest(member=member):
                buffer = io.BytesIO()
                image.save(buffer, "PNG")
                members = {**self.members, member: buffer.getvalue()}
                write_archive(self.path, self.manifest, members)
                with self.assertRaises(ValueError):
                    load_project(self.path)


class ProjectToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_save_close_open_and_failed_load_preserve_sessions(self):
        sessions = SessionManager()
        canvas = sessions.replace("source", sample_canvas())
        before = snapshot(canvas)
        history_size = len(canvas.undo_stack)
        with tempfile.TemporaryDirectory() as directory, patch.object(server, "sessions", sessions):
            path = str(Path(directory) / "project.mcpproj")
            out = await server.app.call_tool("save_project", {"path": path, "session_id": "source"})
            self.assertEqual(json.loads(out[0].text)["layer_count"], 2)
            self.assertEqual(snapshot(canvas), before)
            self.assertEqual(len(canvas.undo_stack), history_size)
            sessions.delete("source")
            out = await server.app.call_tool("open_project", {"path": path, "session_id": "restored"})
            self.assertEqual(json.loads(out[0].text)["layer_count"], 2)
            restored = sessions.get("restored")
            self.assertEqual(snapshot(restored), before)
            Path(path).write_bytes(b"not a project")
            for session_id in ("restored", "missing"):
                out = await server.app.call_tool("open_project", {"path": path, "session_id": session_id})
                self.assertIn("Error opening project", out[0].text)
            self.assertIs(sessions.get("restored"), restored)
            self.assertEqual(snapshot(restored), before)
            self.assertIsNone(sessions.get("missing"))
            out = await server.app.call_tool("save_project", {"path": path, "session_id": "missing"})
            self.assertIn("Error saving project", out[0].text)
            self.assertEqual(Path(path).read_bytes(), b"not a project")
