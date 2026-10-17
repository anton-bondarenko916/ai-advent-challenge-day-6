"""Тесты agentic loop с файловыми инструментами (ТЗ ai_1_35).

Проверяются: LLM сама выбирает инструменты fs-сервера в обычном ходе
чата (без отдельных команд), оба сценария ТЗ («найти использования
компонента» и «сгенерировать файл») на РЕАЛЬНЫХ инструментах
fs_server.py, лимит итераций (10 раундов — защита от зацикливания,
частичный результат с честным сообщением), передача tools fs-сервера
в запрос LLM, лог шагов в UI (tools_panel: номера, иконки, обрезка).

Запуск из корня проекта:
    python -m unittest tests.test_fs_agent -v
"""

from __future__ import annotations

import flet as ft
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import fs_server
from app.ui.chat_view import ChatView
from app.ui.components import (
    FS_SUMMARY_MAX_FILES,
    FS_SUMMARY_MAX_HITS,
    fs_summary_controls,
    tools_panel,
    tool_icon,
)
from src.agent import DeepSeekAgent
from src.tokens import HeuristicCounter
from tests.test_fs_server import _make_fixture


# ----------------------------------------------------------------------
# Заглушки: OpenAI-клиент с очередью ответов + fs-клиент над реальными
# инструментами fs_server
# ----------------------------------------------------------------------
class _FakeCompletions:
    """chat.completions.create — заглушка с очередью ответов."""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.responses:
            return self.responses.pop(0)
        # Очередь пуста: LLM «залипла» на tool_calls (тест лимита)
        return _api_response(_api_message(content=None, tool_calls=[
            _tool_call(f"c{len(self.calls)}", "list_directory",
                       {"path": "."}),
        ]), finish_reason="tool_calls")


class _FakeOpenAIClient:
    def __init__(self, responses=None):
        self.completions = _FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)

    @property
    def calls(self):
        return self.completions.calls


def _api_message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls,
                           reasoning_content=None)


def _api_response(message, finish_reason="stop"):
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None, model="test-model")


def _tool_call(call_id, name, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class FsToolClient:
    """MCP-клиент fs-сервера: вызовы уходят РЕАЛЬНЫМ инструментам
    fs_server (ALLOWED_ROOT подменён на фикстуру) — end-to-end без HTTP."""

    FS_TOOLS = ("read_file", "read_multiple_files", "write_file",
                "search_files", "search_within_files", "list_directory",
                "tree")

    def __init__(self):
        self.calls = []

    async def list_tools(self):
        schemas = {
            "read_file": {"path": {"type": "string"}},
            "read_multiple_files": {"paths": {"type": "array"}},
            "write_file": {"path": {"type": "string"},
                           "content": {"type": "string"}},
            "search_files": {"pattern": {"type": "string"}},
            "search_within_files": {"substring": {"type": "string"}},
            "list_directory": {"path": {"type": "string"}},
            "tree": {"path": {"type": "string"}},
        }
        from src.mcp_models import MCPTool
        return [
            MCPTool(name=name, description=f"Инструмент {name}",
                    input_schema={"type": "object",
                                  "properties": schemas[name]})
            for name in self.FS_TOOLS
        ]

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, dict(arguments or {})))
        dispatch = {
            "read_file": fs_server.read_file,
            "write_file": fs_server.write_file,
            "read_multiple_files": fs_server.read_multiple_files,
            "search_files": fs_server.search_files,
            "search_within_files": fs_server.search_within_files,
            "list_directory": fs_server.list_directory,
            "tree": fs_server.tree,
        }
        return json.dumps(dispatch[name](**(arguments or {})),
                          ensure_ascii=False)


def make_fs_agent(responses, max_tool_rounds=10):
    """DeepSeekAgent с фейковым LLM и fs-клиентом над реальными
    инструментами (корень — фикстура во временном каталоге).

    Returns:
        (agent, api, patcher, tmp): tmp — корень фикстуры (пути файлов
        проверять только ДО patcher.stop()).
    """
    tmp = Path(tempfile.mkdtemp())
    _make_fixture(tmp)
    patcher = mock.patch.object(fs_server, "ALLOWED_ROOT", tmp)
    patcher.start()
    api = _FakeOpenAIClient(responses)
    agent = DeepSeekAgent(
        api_key="test-key", mock_mode=False,
        token_counter=HeuristicCounter(),
        fs_client=FsToolClient(),
        max_tool_rounds=max_tool_rounds,
    )
    agent._client = api
    return agent, api, patcher, tmp


class AgentScenario1FindUsagesTest(unittest.TestCase):
    """Сценарий ТЗ 1: «Найди все места использования DeepSeekAgent»."""

    def tearDown(self) -> None:
        pass

    def test_search_then_read_then_answer(self) -> None:
        agent, api, patcher, _tmp = make_fs_agent([
            # Шаг 1: LLM решает — нужен поиск по содержимому
            _api_response(_api_message(content=None, tool_calls=[
                _tool_call("c1", "search_within_files",
                           {"substring": "DeepSeekAgent",
                            "directory": "src"}),
            ]), finish_reason="tool_calls"),
            # Шаг 2: LLM решает — прочитать найденные файлы
            _api_response(_api_message(content=None, tool_calls=[
                _tool_call("c2", "read_multiple_files",
                           {"paths": ["src/agent.py", "src/main.py"]}),
            ]), finish_reason="tool_calls"),
            # Финальный ответ
            _api_response(_api_message(
                content="Класс DeepSeekAgent используется в 2 файлах: "
                        "src/agent.py и src/main.py.",
            )),
        ])
        try:
            result = agent.send_message(
                "Найди все места, где используется класс DeepSeekAgent")
        finally:
            patcher.stop()

        self.assertIsNone(result["error"])
        self.assertIn("2 файлах", result["text"])
        # Оба шага выполнены на реальных инструментах fs-сервера
        fs = agent._fs_client
        self.assertEqual(fs.calls[0][0], "search_within_files")
        self.assertEqual(fs.calls[1][0], "read_multiple_files")
        self.assertEqual(fs.calls[1][1],
                         {"paths": ["src/agent.py", "src/main.py"]})
        # Лог шагов для UI: 2 события с именем/аргументами/результатом
        steps = result["tool_calls"]
        self.assertEqual([step["name"] for step in steps],
                         ["search_within_files", "read_multiple_files"])
        self.assertTrue(all("result" in step for step in steps))
        # fs-инструменты переданы LLM в первом запросе
        tool_names = {t["function"]["name"]
                      for t in api.calls[0]["tools"]}
        self.assertIn("search_within_files", tool_names)
        self.assertIn("write_file", tool_names)


class AgentScenario2GenerateFileTest(unittest.TestCase):
    """Сценарий ТЗ 2: «Сгенерируй ADR» (навигация → шаблон → запись)."""

    def test_list_read_write_then_answer(self) -> None:
        agent, api, patcher, tmp = make_fs_agent([
            # Шаг 1: посмотреть структуру docs/
            _api_response(_api_message(content=None, tool_calls=[
                _tool_call("c1", "list_directory",
                           {"path": "docs/adr"}),
            ]), finish_reason="tool_calls"),
            # Шаг 2: прочитать существующий ADR как шаблон
            _api_response(_api_message(content=None, tool_calls=[
                _tool_call("c2", "read_file",
                           {"path": "docs/adr/0001-example.md"}),
            ]), finish_reason="tool_calls"),
            # Шаг 3: создать новый файл
            _api_response(_api_message(content=None, tool_calls=[
                _tool_call("c3", "write_file", {
                    "path": "docs/adr/0002-ollama-for-local-models.md",
                    "content": "# ADR-0002: Ollama для локальных моделей\n"
                               "\nКонтекст: нужен офлайн-инференс.\n",
                }),
            ]), finish_reason="tool_calls"),
            _api_response(_api_message(
                content="Создал ADR: docs/adr/0002-ollama-for-local-models.md",
            )),
        ])
        try:
            result = agent.send_message(
                "Сгенерируй ADR о выборе Ollama для локальных моделей")
        finally:
            patcher.stop()

        self.assertIsNone(result["error"])
        self.assertIn("Создал ADR", result["text"])
        # Файл РЕАЛЬНО создан на диске (в корне фикстуры)
        created = tmp / "docs/adr/0002-ollama-for-local-models.md"
        self.assertTrue(created.exists())
        self.assertIn("Ollama", created.read_text(encoding="utf-8"))
        # Путь созданного файла — в результате хода (кнопка «Открыть файл»)
        self.assertEqual(result.get("file_path"), str(created))
        # Цепочка из трёх файловых операций
        fs = agent._fs_client
        self.assertEqual([name for name, _ in fs.calls],
                         ["list_directory", "read_file", "write_file"])


class AgentRoundLimitTest(unittest.TestCase):
    """Лимит итераций (ТЗ: 10) — защита от зацикливания."""

    def test_loop_stops_after_ten_rounds(self) -> None:
        agent, api, patcher, _tmp = make_fs_agent(
            responses=[], max_tool_rounds=10)
        try:
            result = agent.send_message("залипший запрос")
        finally:
            patcher.stop()

        # 1 начальный вызов + 10 раундов с инструментами
        self.assertEqual(len(api.calls), 11)
        self.assertEqual(len(agent._fs_client.calls), 10)
        # Честное сообщение вместо пустого ответа
        self.assertIn("лимит итераций", result["text"])
        self.assertEqual(len(result["tool_calls"]), 10)


class AgentTruncationGuardTest(unittest.TestCase):
    """Обрезка ответа (finish_reason=length) на thinking-модели.

    Прецедент: запрос 325K символов — бюджет max_tokens (1000) уходил в
    reasoning, раунды возвращали ПУСТОЙ контент с tool_calls, цикл
    крутился (7 одинаковых write_file), финальный ответ пуст.
    """

    def test_truncated_tool_calls_retried_with_double_budget(self) -> None:
        """Обрезанный ответ с tool_calls НЕ исполняется: ретрай ×2 бюджета."""
        agent, api, patcher, tmp = make_fs_agent([
            # Раунд 1: нормальный вызов инструмента
            _api_response(_api_message(content=None, tool_calls=[
                _tool_call("c1", "list_directory", {"path": "."}),
            ]), finish_reason="tool_calls"),
            # Раунд 2: ОБРЕЗАН (бюджет ушёл в reasoning), но tool_calls есть
            _api_response(_api_message(content=None, tool_calls=[
                _tool_call("c2", "write_file",
                           {"path": "a.md", "content": "x"}),
            ]), finish_reason="length"),
            # Ретрай с удвоенным бюджетом: модель завершает ответ
            _api_response(_api_message(content="Файл готов."),
                          finish_reason="stop"),
        ])
        try:
            result = agent.send_message("сделай файл")
        finally:
            patcher.stop()

        # Обрезанные tool_calls не исполнялись: только list_directory
        self.assertEqual([name for name, _ in agent._fs_client.calls],
                         ["list_directory"])
        # Ходы с инструментами получают минимум 4000 (thinking-модель
        # тратит бюджет на reasoning), ретрай — удвоенный (8000)
        self.assertEqual(api.calls[1]["max_tokens"], 4000)
        self.assertEqual(api.calls[2]["max_tokens"], 8000)
        self.assertEqual(result["text"], "Файл готов.")

    def test_persistent_truncation_stops_with_honest_message(self) -> None:
        """Ретрай тоже обрезан — цикл останавливается, ответ не пустой."""
        class _AlwaysTruncated:
            """После очереди отвечает ТОЛЬКО обрезанными tool_calls."""

            def __init__(self, first_responses):
                self.responses = list(first_responses)
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                if self.responses:
                    return self.responses.pop(0)
                return _api_response(_api_message(content=None, tool_calls=[
                    _tool_call("ct", "list_directory", {"path": "."}),
                ]), finish_reason="length")

        completions = _AlwaysTruncated([
            _api_response(_api_message(content=None, tool_calls=[
                _tool_call("c1", "list_directory", {"path": "."}),
            ]), finish_reason="tool_calls"),
        ])
        agent, api, patcher, _tmp = make_fs_agent([])
        agent._client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions))
        api = completions
        try:
            result = agent.send_message("тяжёлый запрос")
        finally:
            patcher.stop()

        # Инструмент исполнен один раз (из нормального ответа)
        self.assertEqual(len(agent._fs_client.calls), 1)
        # Ровно один ретрай с удвоенным бюджетом (4000 → 8000)
        self.assertEqual(len(api.calls), 3)
        self.assertEqual(api.calls[2]["max_tokens"], 8000)
        # Честное сообщение вместо «(пустой ответ)»
        self.assertIn("не вернула текст", result["text"])


class ToolStepsPanelTest(unittest.TestCase):
    """Лог шагов в UI: номера, иконки, обрезка результата, скролл."""

    def test_tool_icon_mapping(self) -> None:
        self.assertEqual(tool_icon("search_within_files"), "🔍")
        self.assertEqual(tool_icon("write_file"), "✍️")
        self.assertEqual(tool_icon("tree"), "🌳")
        self.assertEqual(tool_icon("неизвестный_инструмент"), "🔧")

    def test_panel_shows_numbered_steps(self) -> None:
        panel = tools_panel([
            {"name": "search_within_files",
             "arguments": {"substring": "X"}, "result": "[]"},
            {"name": "write_file",
             "arguments": {"path": "a.md"}, "result": "Файл создан: a.md"},
        ])
        self.assertEqual(panel.title.value, "🛠 Инструменты (2)")
        rows = panel.controls[0].controls  # шаги внутри скролл-Column
        self.assertIn("Шаг 1:", rows[0].value)
        self.assertIn("🔍 search_within_files", rows[0].value)
        self.assertIn("Шаг 2:", rows[1].value)
        self.assertIn("✍️ write_file", rows[1].value)

    def test_panel_long_result_truncated(self) -> None:
        from app.ui.components import PREVIEW_TOOL_RESULT

        long_result = "y" * (PREVIEW_TOOL_RESULT * 3)
        panel = tools_panel([
            {"name": "read_file", "arguments": {"path": "big.md"},
             "result": long_result},
        ])
        self.assertLess(len(panel.controls[0].controls[0].value),
                        PREVIEW_TOOL_RESULT + 100)

    def test_panel_is_scrollable_with_bounded_height(self) -> None:
        """Скролл: длинный лог не растягивает чат (ТЗ ai_1_35)."""
        steps = [{"name": "read_file", "arguments": {"path": f"f{i}.py"},
                  "result": "x" * 200} for i in range(12)]
        panel = tools_panel(steps)
        column = panel.controls[0]
        self.assertEqual(column.scroll, ft.ScrollMode.AUTO)
        self.assertLessEqual(column.height, 340)
        self.assertEqual(len(column.controls), 12)

    def test_panel_empty_is_none(self) -> None:
        self.assertIsNone(tools_panel([]))


class FsSummaryControlsTest(unittest.TestCase):
    """Сводка файловых операций в самом сообщении (ТЗ ai_1_35)."""

    def test_search_within_summary_with_wrapper(self) -> None:
        """Формат живого хода: FastMCP-обёртка {"result": [...]}."""
        hits = [
            {"file": "src/agent.py", "line": 1, "content": "class DeepSeekAgent:"},
            {"file": "src/main.py", "line": 1, "content": "from agent import DeepSeekAgent"},
            {"file": "src/main.py", "line": 2, "content": "agent = DeepSeekAgent()"},
        ]
        controls = fs_summary_controls([{
            "name": "search_within_files",
            "arguments": {"substring": "DeepSeekAgent", "directory": "src"},
            "result": json.dumps({"result": hits}, ensure_ascii=False),
        }])
        self.assertEqual(len(controls), 1)
        text = controls[0].value
        self.assertIn("🔍 Поиск «DeepSeekAgent» (в src)", text)
        self.assertIn("найдено 3 вхождения в 2 файлах", text)
        self.assertIn("• src/agent.py:1 — class DeepSeekAgent:", text)
        self.assertIn("• src/main.py:2", text)

    def test_search_within_summary_raw_list_and_cap(self) -> None:
        """Офлайн-формат (сырой JSON) + предел записей в сообщении."""
        hits = [{"file": f"f{i}.py", "line": 1, "content": "X"}
                for i in range(FS_SUMMARY_MAX_HITS + 4)]
        controls = fs_summary_controls([{
            "name": "search_within_files",
            "arguments": {"substring": "X"},
            "result": json.dumps(hits),
        }])
        text = controls[0].value
        self.assertIn("найдено 12 вхождений в 12 файлах", text)
        self.assertEqual(text.count("   • "), FS_SUMMARY_MAX_HITS)
        self.assertIn("ещё 4 вхождения", text)

    def test_search_files_summary(self) -> None:
        paths = ["docs/a.md", "docs/b.md", "docs/c.md"]
        controls = fs_summary_controls([{
            "name": "search_files",
            "arguments": {"pattern": "*.md", "directory": "docs"},
            "result": json.dumps(paths),
        }])
        text = controls[0].value
        self.assertIn("🔍 Файлы по шаблону «*.md»: найдено 3", text)
        self.assertIn("• docs/a.md", text)

    def test_search_files_list_cap(self) -> None:
        paths = [f"d{i}.md" for i in range(FS_SUMMARY_MAX_FILES + 5)]
        controls = fs_summary_controls([{
            "name": "search_files", "arguments": {"pattern": "*.md"},
            "result": json.dumps(paths),
        }])
        text = controls[0].value
        self.assertEqual(text.count("   • "), FS_SUMMARY_MAX_FILES)
        self.assertIn("ещё 5 файлов", text)

    def test_write_file_summary_shows_saved_path(self) -> None:
        """ТЗ: путь сохранения — в самом сообщении."""
        controls = fs_summary_controls([{
            "name": "write_file",
            "arguments": {"path": "docs/adr/0002.md"},
            "result": json.dumps(
                {"result": r"Файл создан: C:\prj\docs\adr\0002.md"}),
        }])
        self.assertEqual(len(controls), 1)
        self.assertEqual(controls[0].value,
                         "✍️ Создан файл: C:\\prj\\docs\\adr\\0002.md")

    def test_error_results_excluded(self) -> None:
        """Ошибки инструментов в сводку не попадают (видны в шагах)."""
        for result in (
            json.dumps({"result": {"error": "Доступ запрещён"}}),
            json.dumps([{"error": "Доступ запрещён"}]),
            json.dumps({"error": "нет такого файла"}),
            "не json",
        ):
            self.assertEqual(fs_summary_controls([{
                "name": "search_within_files",
                "arguments": {"substring": "X"},
                "result": result,
            }]), [], result)
        # Ошибка записи тоже не попадает
        self.assertEqual(fs_summary_controls([{
            "name": "write_file", "arguments": {"path": "x"},
            "result": json.dumps({"result": '{"error": "отказ"}'}),
        }]), [])

    def test_search_flood_capped_to_three_blocks(self) -> None:
        """Пять поисков подряд → 3 блока в сообщении + пометка (поток
        блоков топит ответ; полные результаты — в шагах)."""
        calls = [{
            "name": "search_within_files",
            "arguments": {"substring": f"q{i}"},
            "result": json.dumps([{"file": "a.py", "line": 1,
                                   "content": f"q{i}"}]),
        } for i in range(5)]
        controls = fs_summary_controls(calls)
        texts = [c.value for c in controls]
        self.assertEqual(len(texts), 4)  # 3 блока + пометка
        self.assertTrue(sum("🔍 Поиск" in t for t in texts) == 3)
        self.assertIn("ещё 2 поисков", texts[-1])

    def test_created_files_not_capped_by_search_limit(self) -> None:
        """Строки созданных файлов не режутся лимитом поисковых блоков."""
        calls = [
            {"name": "search_within_files", "arguments": {"substring": "x"},
             "result": json.dumps([{"file": "a.py", "line": 1,
                                    "content": "x"}])},
            {"name": "write_file", "arguments": {"path": "a.md"},
             "result": json.dumps({"result": "Файл создан: /p/a.md"})},
            {"name": "write_file", "arguments": {"path": "b.md"},
             "result": json.dumps({"result": "Файл создан: /p/b.md"})},
        ]
        texts = [c.value for c in fs_summary_controls(calls)]
        self.assertEqual(len(texts), 3)
        self.assertTrue(any("/p/a.md" in t for t in texts))
        self.assertTrue(any("/p/b.md" in t for t in texts))

    def test_non_fs_tools_ignored(self) -> None:
        controls = fs_summary_controls([
            {"name": "get_weather", "arguments": {"city": "Berlin"},
             "result": json.dumps({"result": "15C"})},
        ])
        self.assertEqual(controls, [])

    def test_mixed_calls_keep_order(self) -> None:
        controls = fs_summary_controls([
            {"name": "search_files", "arguments": {"pattern": "*.md"},
             "result": json.dumps(["docs/a.md"])},
            {"name": "write_file", "arguments": {"path": "docs/b.md"},
             "result": json.dumps({"result": "Файл создан: /prj/docs/b.md"})},
        ])
        self.assertEqual(len(controls), 2)
        self.assertIn("Файлы по шаблону", controls[0].value)
        self.assertIn("Создан файл", controls[1].value)


class ChatViewFsSummaryTest(unittest.TestCase):
    """Сводка попадает в пузырь ответа; кнопка «Открыть файл» — по пути."""

    def test_answer_contains_fs_summary_and_open_button(self) -> None:
        from tests.test_support_ui import make_app
        from tests.test_app_agent import make_agent

        view = ChatView(make_app(make_agent()))
        saved = r"C:\prj\docs\adr\0002.md"
        view.show_thinking()
        view.add_assistant_result({
            "text": "Создал ADR.",
            "tool_calls": [{
                "name": "write_file", "arguments": {"path": "docs/adr/0002.md"},
                "result": json.dumps(
                    {"result": f"Файл создан: {saved}"}),
            }],
            "file_path": saved,
        })
        bubble = view.messages_list.controls[-1]
        texts = [control.value for control in bubble.content.controls
                 if hasattr(control, "value") and control.value]
        self.assertTrue(any("✍️ Создан файл" in text for text in texts))
        # Кнопка «Открыть файл» знает сохранённый путь
        tooltips = [getattr(control, "tooltip", "")
                    for control in bubble.content.controls]
        self.assertIn(saved, tooltips)


if __name__ == "__main__":
    unittest.main()
