import json
import unittest

from session import SessionManager


class SessionManagerTests(unittest.TestCase):
    def setUp(self):
        self.sessions = SessionManager()

    def test_list_sessions_empty_until_used(self):
        self.assertEqual(self.sessions.list_sessions(), {})
        self.sessions.get_default_session()
        self.assertIn("default", self.sessions.list_sessions())

    def test_sessions_are_isolated_documents(self):
        a = self.sessions.create("a", 64, 64, (1, 2, 3))
        b = self.sessions.create("b", 32, 32, (9, 9, 9))
        a.add_layer(image=None)
        self.assertEqual((a.width, a.height), (64, 64))
        self.assertEqual((b.width, b.height), (32, 32))
        self.assertEqual(len(a.layers), 2)
        self.assertEqual(len(b.layers), 1)

    def test_get_or_create_reuses_existing(self):
        first = self.sessions.get_or_create("x")
        second = self.sessions.get_or_create("x")
        self.assertIs(first, second)
        self.assertEqual(set(self.sessions.list_sessions()), {"x"})

    def test_close_session(self):
        self.sessions.create("temp", 16, 16)
        self.assertTrue(self.sessions.delete("temp"))
        self.assertFalse(self.sessions.delete("temp"))


class SessionToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import server
        self.server = server
        for sid in list(server.sessions.list_sessions()):
            server.sessions.delete(sid)

    async def test_list_and_close_session_tools(self):
        await self.server.new_canvas(width=64, height=64, session_id="photo")
        report = json.loads((await self.server.list_sessions_tool())[0].text)
        self.assertEqual(report["photo"]["size"], [64, 64])
        self.assertEqual(report["photo"]["layers"], 1)
        out = await self.server.close_session_tool(session_id="photo")
        self.assertIn("closed", out[0].text)
        report = json.loads((await self.server.list_sessions_tool())[0].text)
        self.assertNotIn("photo", report)
        out = await self.server.close_session_tool(session_id="photo")
        self.assertIn("No open session", out[0].text)

    async def test_canvas_tools_address_session_by_id(self):
        await self.server.new_canvas(width=64, height=64, session_id="s1")
        await self.server.new_canvas(width=32, height=32, session_id="s2")
        await self.server.adjust_tool(brightness=0.5, session_id="s1")
        info_s1 = json.loads((await self.server.get_info(session_id="s1"))[0].text)
        info_s2 = json.loads((await self.server.get_info(session_id="s2"))[0].text)
        self.assertEqual(info_s1["width"], 64)
        self.assertEqual(info_s2["width"], 32)
        self.assertTrue(info_s1["undo_available"])    # adjust pushed a state
        self.assertFalse(info_s2["undo_available"])   # untouched document

    async def test_preview_and_editing_tools_address_session_by_id(self):
        from editing import register_editing_tools

        class Registry:
            def __init__(self):
                self.tools = {}

            def tool(self, name):
                def register(function):
                    self.tools[name] = function
                    return function
                return register

        registry = Registry()
        run = unittest.mock.AsyncMock(return_value=None)
        register_editing_tools(registry, unittest.mock.AsyncMock(), self.server.sessions, run)
        await self.server.new_canvas(width=48, height=48, session_id="edit")
        out = await registry.tools["preview_canvas"](session_id="edit")
        self.assertEqual(json.loads(out[0].text)["width"], 48)
        out = await registry.tools["preview_canvas"]()  # default session, independent
        self.assertNotEqual(json.loads(out[0].text)["width"], 48)


if __name__ == "__main__":
    unittest.main()
