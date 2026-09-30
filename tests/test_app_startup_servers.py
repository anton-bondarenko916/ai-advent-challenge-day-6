"""Тесты старта Flet-приложения: автозапуск MCP-серверов, ленивые импорты.

Регрессии запуска (жалоба пользователя, ai_1_22-fix): приложение
стартовало ~9 с (langchain_text_splitters при импорте тянет
transformers) и MCP-серверы поднимались только после первого хода LLM
через self-healing хук.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import launch_mcp_servers  # noqa: E402


class _FakeLauncher:
    """Лаунчер-заглушка (без спавна процессов)."""

    def __init__(self, label: str = "Фейк", fail: bool = False) -> None:
        self.label = label
        self.fail = fail
        self.calls = 0

    def launch(self):
        self.calls += 1
        if self.fail:
            raise OSError("нет процесса")
        return True, "запущен"


class LaunchMcpServersTest(unittest.TestCase):
    """Автозапуск серверов при старте Flet-версии."""

    def _run(self, fail_labels=()):
        """Зпатчит настройки и лаунчеры, возвращает (messages, instances).

        reminder и geo включены, news и weather выключены — проверяет
        фильтрацию по enabled.
        """
        settings = {
            "get_reminder_settings": {"enabled": True,
                                      "host": "127.0.0.1", "port": 8000},
            "get_news_settings": {"enabled": False,
                                  "host": "127.0.0.1", "port": 8001},
            "get_geo_settings": {"enabled": True,
                                 "host": "127.0.0.1", "port": 8002},
            "get_weather_settings": {"enabled": False,
                                     "host": "127.0.0.1", "port": 8003},
        }
        classes = {
            "src.reminder_launcher.ReminderServerLauncher": "Reminder",
            "src.news_launcher.NewsServerLauncher": "News",
            "src.geo_launcher.GeoServerLauncher": "Geo",
            "src.weather_launcher.WeatherServerLauncher": "Weather",
        }
        instances = {
            name: _FakeLauncher(label, fail=(label in fail_labels))
            for name, label in classes.items()
        }
        import app.main as app_main
        # Замена — функция, возвращающая настройки (как get_*_settings)
        stubs = {name: (lambda cfg, data=value: data)
                 for name, value in settings.items()}
        with mock.patch.multiple(app_main, **stubs):
            patches = [
                mock.patch(name, return_value=instances[name])
                for name in classes
            ]
            for patcher in patches:
                patcher.start()
                self.addCleanup(patcher.stop)
            return launch_mcp_servers({}), instances

    def test_launches_enabled_servers(self) -> None:
        messages, instances = self._run()
        # Включены reminder и geo (news/weather выключены — не запускаются)
        self.assertEqual(len(messages), 2, messages)
        self.assertIn("Reminder: запущен", messages)
        self.assertIn("Geo: запущен", messages)
        self.assertEqual(
            instances["src.reminder_launcher.ReminderServerLauncher"].calls, 1)
        self.assertEqual(
            instances["src.news_launcher.NewsServerLauncher"].calls, 0)

    def test_error_does_not_break_startup(self) -> None:
        messages, _ = self._run(fail_labels=("Reminder",))
        self.assertIn("Ошибка автозапуска", messages[0])
        # Geo всё равно запущен (сбой одного не останавливает остальные)
        self.assertIn("Geo: запущен", messages[1])


class LazyChunkersImportTest(unittest.TestCase):
    """langchain_text_splitters не импортируется при старте приложения.

    Регрессия: module-level импорт в app.rag.chunkers тянул
    sentence_transformers/transformers (~7.5 с) — окно открывалось
    поздно. Сплиттеры нужны только в момент индексации.
    """

    def test_chunkers_import_is_light(self) -> None:
        code = (
            "import sys;"
            "import app.rag.chunkers;"
            "heavy = [m for m in sys.modules"
            " if m.startswith(('langchain_text_splitters',"
            " 'sentence_transformers', 'transformers'))];"
            "print('HEAVY:', heavy)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True,
            text=True, timeout=120, cwd=str(PROJECT_ROOT))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("HEAVY: []", result.stdout, result.stdout)

    def test_chunking_still_works_after_lazy_import(self) -> None:
        from app.rag.chunkers import chunk_by_strategy
        documents = chunk_by_strategy("# Заголовок\n\nТекст документа",
                                      "structure", chunk_size=100,
                                      overlap=10)
        self.assertGreaterEqual(len(documents), 1)


if __name__ == "__main__":
    unittest.main()
