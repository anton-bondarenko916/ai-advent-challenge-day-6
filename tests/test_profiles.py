"""Тесты персонализации: профили пользователя (третий слой памяти).

Проверяют механизм персонализации end-to-end (без реального API —
поведенческие различия стилей обеспечивает сам системный промпт):

- модели профиля: сериализация/разбор constraints и context;
- сборщик системного промпта: порядок секций, правила приоритета,
  разные профили дают разные промпты (Тест 1), учёт constraints
  без явного упоминания (Тест 2), правила приоритета в промпте (Тест 3);
- ProfileRepository: миграция (таблицы, профиль Default из .env,
  привязка старых сообщений — Тест 7), CRUD, настройки агента,
  каскадное удаление истории (Тест 6);
- DeepSeekAgent: системный промпт из профиля, изоляция истории по
  профилям (Тест 4), применение настроек «на лету» (Тест 5);
- GUI: вкладки «Профиль» и «Настройки агента», переключатель профиля
  в статус-баре (без дисплея — аккуратно пропускаются).

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from src.models import (
    ProfileAgentSettings,
    UserProfile,
    parse_constraints,
    parse_context,
    parse_context_lines,
    serialize_constraints,
    serialize_context,
)
from src.profile_prompt_builder import build_system_prompt
from src.profile_repository import DEFAULT_PROFILE_NAME, ProfileRepository
from src.storage import ContextStore
from src.tokens import HeuristicCounter
from tests import gui_helpers

_TESTS_TMP = Path(__file__).resolve().parent / "tmp_profiles"


def _config(**overrides) -> dict:
    """Конфигурация (.env-совместимая) для профиля Default при миграции."""
    config = {
        "DEFAULT_MODEL": "deepseek-v4-flash",
        "DEFAULT_TEMPERATURE": 0.7,
        "DEFAULT_MAX_TOKENS": 1000,
        "DEFAULT_SYSTEM_PROMPT": "Ты полезный AI-ассистент.",
    }
    config.update(overrides)
    return config


def _make_repo(name: str, config: dict | None = None) -> ProfileRepository:
    """Репозиторий профилей на временной БД."""
    return ProfileRepository(
        db_path=_TESTS_TMP / f"{name}_{uuid.uuid4().hex}.db",
        config=config or _config(),
    )


# ---------------------------------------------------------------------------
# Доменные модели: сериализация сложных полей
# ---------------------------------------------------------------------------

class ModelsTest(unittest.TestCase):
    """Сериализация constraints/context и разбор многострочного текста."""

    def test_serialize_and_parse_constraints(self) -> None:
        raw = serialize_constraints(["только Python", "без эмодзи"])
        self.assertEqual(
            parse_constraints(raw), ["только Python", "без эмодзи"]
        )

    def test_parse_constraints_garbage_safe(self) -> None:
        self.assertEqual(parse_constraints(None), [])
        self.assertEqual(parse_constraints(""), [])
        self.assertEqual(parse_constraints("{битый json"), [])
        self.assertEqual(parse_constraints('["a", 1]'), ["a", "1"])

    def test_serialize_and_parse_context(self) -> None:
        raw = serialize_context({"role": "разработчик", "stack": ["Python"]})
        self.assertEqual(
            parse_context(raw),
            {"role": "разработчик", "stack": ["Python"]},
        )

    def test_parse_context_garbage_safe(self) -> None:
        self.assertEqual(parse_context(None), {})
        self.assertEqual(parse_context("[1, 2]"), {})

    def test_parse_context_lines(self) -> None:
        constraints, context = parse_context_lines(
            'не использовать эмодзи\nrole: разработчик\n'
            'stack: ["Python", "SQL"]'
        )
        self.assertEqual(constraints, ["не использовать эмодзи"])
        self.assertEqual(
            context, {"role": "разработчик", "stack": ["Python", "SQL"]}
        )


# ---------------------------------------------------------------------------
# Сборщик системного промпта
# ---------------------------------------------------------------------------

class PromptBuilderTest(unittest.TestCase):
    """build_system_prompt: порядок секций и правила приоритета."""

    def test_full_prompt_sections_in_order(self) -> None:
        profile = UserProfile(
            name="Senior Dev", display_name="Антон",
            style="краткий, технический",
            format="markdown, код в блоках", language="русский",
            constraints=["не использовать эмодзи", "только Python"],
            context={"role": "разработчик", "stack": ["Python", "SQL"]},
        )
        settings = ProfileAgentSettings(
            profile_id=1, base_system_prompt="Базовая инструкция."
        )
        prompt = build_system_prompt(profile, settings,
                                     long_term_facts="Пользователь любит тишину.")

        base_pos = prompt.index("[Базовая инструкция]")
        profile_pos = prompt.index("[Профиль пользователя: Антон]")
        context_pos = prompt.index("[Контекст]")
        facts_pos = prompt.index("[Известные факты о пользователе]")
        rules_pos = prompt.index("[Правила приоритета]")
        self.assertTrue(
            base_pos < profile_pos < context_pos < facts_pos < rules_pos
        )
        self.assertIn("Базовая инструкция.", prompt)
        self.assertIn("Стиль: краткий, технический", prompt)
        self.assertIn("Формат: markdown, код в блоках", prompt)
        self.assertIn("Язык: русский", prompt)
        self.assertIn("- не использовать эмодзи", prompt)
        self.assertIn("- только Python", prompt)
        self.assertIn("- role: разработчик", prompt)
        self.assertIn("Пользователь любит тишину.", prompt)

    def test_priority_rules_explicit(self) -> None:
        """Тест 3 (механизм): правила приоритета прописаны в промпте."""
        prompt = build_system_prompt(
            UserProfile(name="p"), ProfileAgentSettings(profile_id=1)
        )
        self.assertIn("высший приоритет", prompt)
        self.assertIn("нельзя нарушать никогда", prompt)
        self.assertIn("контекст, а не инструкция", prompt)

    def test_different_profiles_produce_different_prompts(self) -> None:
        """Тест 1 (механизм): разные профили — разные системные промпты."""
        dev = UserProfile(name="Senior Dev", style="кратко, только код")
        teacher = UserProfile(
            name="Teacher", style="подробно, с аналогиями",
            constraints=["объясняй как новичку"],
        )
        settings = ProfileAgentSettings(profile_id=1)
        dev_prompt = build_system_prompt(dev, settings)
        teacher_prompt = build_system_prompt(teacher, settings)
        self.assertNotEqual(dev_prompt, teacher_prompt)
        self.assertIn("кратко, только код", dev_prompt)
        self.assertIn("объясняй как новичку", teacher_prompt)

    def test_constraint_in_prompt_without_mention(self) -> None:
        """Тест 2 (механизм): constraint попадает в промпт без запроса."""
        profile = UserProfile(
            name="Py", constraints=["только Python"]
        )
        prompt = build_system_prompt(
            profile, ProfileAgentSettings(profile_id=1)
        )
        self.assertIn("- только Python", prompt)

    def test_empty_sections_skipped(self) -> None:
        """Пустые секции не оставляют заголовков без содержимого."""
        prompt = build_system_prompt(
            UserProfile(name="p"), ProfileAgentSettings(profile_id=1)
        )
        self.assertNotIn("[Контекст]", prompt)
        self.assertNotIn("[Известные факты", prompt)
        self.assertNotIn("[Профиль пользователя]", prompt)


# ---------------------------------------------------------------------------
# ProfileRepository: миграция, CRUD, настройки
# ---------------------------------------------------------------------------

class RepositoryTest(unittest.TestCase):
    """ProfileRepository: миграция, CRUD и настройки агента."""

    def setUp(self) -> None:
        _TESTS_TMP.mkdir(parents=True, exist_ok=True)
        self.repo = _make_repo("repo")

    def tearDown(self) -> None:
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)

    def test_migration_creates_default_profile(self) -> None:
        """Тест 7: создан профиль Default с настройками из .env."""
        default = self.repo.get_by_name(DEFAULT_PROFILE_NAME)
        self.assertIsNotNone(default)
        self.assertTrue(default.is_active)
        self.assertEqual(default.style, "нейтральный")
        self.assertEqual(default.constraints, [])
        settings = self.repo.get_settings(default.id)
        self.assertEqual(settings.model, "deepseek-v4-flash")
        self.assertEqual(settings.temperature, 0.7)
        self.assertEqual(settings.max_tokens, 1000)
        self.assertEqual(settings.base_system_prompt, "Ты полезный AI-ассистент.")

    def test_migration_is_idempotent(self) -> None:
        """Повторная миграция не создаёт второй Default."""
        self.assertTrue(self.repo.migrate())
        self.assertEqual(len(self.repo.get_all()), 1)

    def test_default_settings_from_env_overrides(self) -> None:
        """Настройки Default берутся из переданной конфигурации (.env)."""
        repo = _make_repo("env", _config(
            DEFAULT_MODEL="deepseek-v4-pro",
            DEFAULT_TEMPERATURE="0.3",
            DEFAULT_MAX_TOKENS="2000",
            DEFAULT_SYSTEM_PROMPT="Особый промпт",
        ))
        settings = repo.get_settings(repo.get_active().id)
        self.assertEqual(settings.model, "deepseek-v4-pro")
        self.assertEqual(settings.temperature, 0.3)
        self.assertEqual(settings.max_tokens, 2000)
        self.assertEqual(settings.base_system_prompt, "Особый промпт")

    def test_crud_cycle(self) -> None:
        created = self.repo.create(UserProfile(
            name="Senior Dev", display_name="Антон", style="кратко",
            constraints=["без эмодзи"], context={"role": "dev"},
        ))
        self.assertIsNotNone(created.id)

        loaded = self.repo.get_by_id(created.id)
        self.assertEqual(loaded.name, "Senior Dev")
        self.assertEqual(loaded.constraints, ["без эмодзи"])
        self.assertEqual(loaded.context, {"role": "dev"})

        loaded.style = "подробно"
        self.repo.update(loaded)
        self.assertEqual(self.repo.get_by_id(created.id).style, "подробно")

        self.assertTrue(self.repo.delete(created.id))
        self.assertIsNone(self.repo.get_by_id(created.id))

    def test_create_duplicate_name_rejected(self) -> None:
        self.repo.create(UserProfile(name="A"))
        with self.assertRaises(ValueError):
            self.repo.create(UserProfile(name="A"))
        with self.assertRaises(ValueError):
            self.repo.create(UserProfile(name="  "))

    def test_update_name_conflict_rejected(self) -> None:
        first = self.repo.create(UserProfile(name="A"))
        self.repo.create(UserProfile(name="B"))
        second = self.repo.get_by_name("B")
        second.name = "A"
        with self.assertRaises(ValueError):
            self.repo.update(second)
        self.repo.update(first)  # своё имя — ок

    def test_set_active_exclusive(self) -> None:
        a = self.repo.create(UserProfile(name="A"))
        b = self.repo.create(UserProfile(name="B"))
        self.assertTrue(self.repo.set_active(b.id))
        self.assertFalse(self.repo.get_by_id(a.id).is_active)
        self.assertTrue(self.repo.get_by_id(b.id).is_active)
        self.assertEqual(self.repo.get_active().id, b.id)

    def test_delete_last_profile_rejected(self) -> None:
        only = self.repo.get_by_name(DEFAULT_PROFILE_NAME)
        self.assertFalse(self.repo.delete(only.id))
        self.assertIsNotNone(self.repo.get_by_id(only.id))

    def test_delete_reactivates_remaining(self) -> None:
        a = self.repo.create(UserProfile(name="A"))
        self.repo.set_active(a.id)
        self.assertTrue(self.repo.delete(a.id))
        self.assertIsNotNone(self.repo.get_active())

    def test_settings_upsert(self) -> None:
        profile = self.repo.create(UserProfile(name="A"))
        settings = self.repo.get_settings(profile.id)
        self.assertEqual(settings.model, "deepseek-v4-flash")
        settings.temperature = 0.2
        settings.max_tokens = 500
        settings.model = "deepseek-v4-pro"
        self.assertTrue(self.repo.update_settings(settings))
        reloaded = self.repo.get_settings(profile.id)
        self.assertEqual(reloaded.temperature, 0.2)
        self.assertEqual(reloaded.max_tokens, 500)
        self.assertEqual(reloaded.model, "deepseek-v4-pro")

    def test_migration_log_file_written(self) -> None:
        """Ошибки и события миграции пишутся в migration.log."""
        from src.config import PROJECT_ROOT
        log_path = PROJECT_ROOT / "migration.log"
        self.assertTrue(log_path.exists())
        content = log_path.read_text(encoding="utf-8", errors="replace")
        self.assertIn("Default", content)


# ---------------------------------------------------------------------------
# Миграция на существующей БД (Тест 7)
# ---------------------------------------------------------------------------

class LegacyMigrationTest(unittest.TestCase):
    """Миграция: новая версия на БД со старыми сообщениями."""

    def setUp(self) -> None:
        _TESTS_TMP.mkdir(parents=True, exist_ok=True)
        self.db_path = _TESTS_TMP / f"legacy_{uuid.uuid4().hex}.db"

    def tearDown(self) -> None:
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)

    def test_old_messages_attached_to_default(self) -> None:
        """Старая БД: сообщения есть, таблиц профилей нет → миграция."""

        # 1. Старая версия приложения: хранилище создало messages,
        #    таблиц профилей ещё не существует.
        storage = ContextStore(db_path=self.db_path, enabled=True)
        storage.save_state(
            [{"role": "system", "content": "старый промпт"},
             {"role": "user", "content": "старый вопрос"},
             {"role": "assistant", "content": "старый ответ"}],
            {"model": "deepseek-v4-flash", "temperature": 0.7,
             "max_tokens": 1000, "system_prompt": "старый промпт"},
        )
        storage.close()
        with sqlite3.connect(self.db_path) as conn:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertIn("messages", tables)
        self.assertNotIn("profiles", tables)

        # 2. Первый запуск новой версии: миграция.
        repo = ProfileRepository(db_path=self.db_path, config=_config(
            DEFAULT_MODEL="deepseek-v4-pro", DEFAULT_TEMPERATURE=0.5,
        ))
        default = repo.get_by_name(DEFAULT_PROFILE_NAME)
        self.assertIsNotNone(default)
        self.assertTrue(default.is_active)
        settings = repo.get_settings(default.id)
        self.assertEqual(settings.model, "deepseek-v4-pro")
        self.assertEqual(settings.temperature, 0.5)

        # 3. Существующие сообщения привязаны к Default.
        self.assertGreaterEqual(repo.count_profile_messages(default.id), 2)
        with sqlite3.connect(self.db_path) as conn:
            orphans = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE profile_id IS NULL"
            ).fetchone()[0]
        self.assertEqual(orphans, 0)

        # 4. Индекс для быстрой выборки по профилю создан.
        with sqlite3.connect(self.db_path) as conn:
            indexes = {row[1] for row in conn.execute(
                "PRAGMA index_list(messages)")}
        self.assertIn("idx_messages_profile", indexes)

    def test_migration_adds_profile_column(self) -> None:
        """ALTER TABLE: колонка profile_id добавляется к старой таблице."""
        storage = ContextStore(db_path=self.db_path, enabled=True)
        storage.save_state(
            [{"role": "system", "content": "p"},
             {"role": "user", "content": "q"}],
            {"model": "deepseek-v4-flash"},
        )
        storage.close()
        repo = ProfileRepository(db_path=self.db_path, config=_config())
        with sqlite3.connect(self.db_path) as conn:
            columns = {row[1] for row in conn.execute(
                "PRAGMA table_info(messages)")}
        self.assertIn("profile_id", columns)
        default = repo.get_by_name(DEFAULT_PROFILE_NAME)
        self.assertEqual(repo.count_profile_messages(default.id), 2)


# ---------------------------------------------------------------------------
# DeepSeekAgent: интеграция с профилем
# ---------------------------------------------------------------------------

class AgentProfileTest(unittest.TestCase):
    """Агент: промпт из профиля, изоляция истории, настройки «на лету»."""

    def setUp(self) -> None:
        _TESTS_TMP.mkdir(parents=True, exist_ok=True)
        self.repo = _make_repo("agent")
        self.agent = self._make_agent()

    def tearDown(self) -> None:
        self.agent.close()
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)

    def _make_agent(self, **kwargs) -> object:
        from src.agent import DeepSeekAgent
        return DeepSeekAgent(
            api_key="k", mock_mode=True, token_counter=HeuristicCounter(),
            profile_repository=self.repo, **kwargs,
        )

    def test_agent_uses_profile_prompt(self) -> None:
        """Системный промпт агента собран из активного профиля."""
        self.assertIn("[Профиль пользователя: Default]", self.agent.system_prompt)
        self.assertIn("[Правила приоритета]", self.agent.system_prompt)

    def test_agent_applies_profile_settings(self) -> None:
        settings = self.repo.get_settings(self.repo.get_active().id)
        settings.temperature = 0.3
        settings.max_tokens = 777
        settings.model = "deepseek-v4-pro"
        self.repo.update_settings(settings)

        agent = self._make_agent()
        try:
            self.assertEqual(agent.temperature, 0.3)
            self.assertEqual(agent.max_tokens, 777)
            self.assertEqual(agent.model, "deepseek-v4-pro")
        finally:
            agent.close()

    def test_profile_switch_replaces_system_message(self) -> None:
        """set_profile: system-сообщение заменяется, дубликатов нет."""
        dev = self.repo.create(UserProfile(
            name="Senior Dev", style="кратко, только код",
        ))
        result = self.agent.set_profile(dev.id)
        self.assertIsNone(result["error"])
        system_messages = [
            m for m in self.agent.history if m["role"] == "system"
        ]
        self.assertEqual(len(system_messages), 1)
        self.assertIn("Senior Dev", system_messages[0]["content"])
        # Профиль зафиксирован активным в БД
        self.assertEqual(self.repo.get_active().id, dev.id)

    def test_history_isolated_between_profiles(self) -> None:
        """Тест 4: у каждого профиля своя история, переключение без потери."""
        other = self.repo.create(UserProfile(name="Профиль B"))

        # Профиль A (Default): диалог про Python
        result = self.agent.send_message("Расскажи про Python")
        self.assertIsNone(result["error"])
        self.assertTrue(any(
            "Python" in m["content"] for m in self.agent.history
        ))

        # Переключение на профиль B: история не содержит Python
        self.agent.set_profile(other.id)
        self.assertFalse(any(
            "Python" in m["content"] for m in self.agent.history
        ))

        # Обратно на A: диалог восстановлен
        self.agent.set_profile(self.repo.get_by_name(DEFAULT_PROFILE_NAME).id)
        self.assertTrue(any(
            "Python" in m["content"] for m in self.agent.history
        ))

    def test_settings_applied_on_the_fly(self) -> None:
        """Тест 5: изменение настроек активного профиля действует без
        перезапуска — send_message перечитывает профиль из БД."""
        result = self.agent.send_message("привет")
        self.assertIsNone(result["error"])

        settings = self.repo.get_settings(self.repo.get_active().id)
        settings.temperature = 0.1
        settings.max_tokens = 333
        self.repo.update_settings(settings)

        result = self.agent.send_message("ещё раз привет")
        self.assertIsNone(result["error"])
        self.assertEqual(self.agent.temperature, 0.1)
        self.assertEqual(self.agent.max_tokens, 333)

    def test_profile_edit_rebuilds_prompt(self) -> None:
        """Редактирование профиля перестраивает system-промпт агента."""
        profile = self.repo.get_active()
        profile.style = "отвечай кратко"
        profile.constraints = ["только Python"]
        self.repo.update(profile)

        result = self.agent.refresh_profile()
        self.assertIsNone(result["error"])
        self.assertIn("отвечай кратко", self.agent.system_prompt)
        self.assertIn("- только Python", self.agent.system_prompt)


class AgentPersistenceTest(unittest.TestCase):
    """Агент + хранилище: сообщения пишутся с profile_id профиля.

    Персист в SQLite выполняется только вне mock-режима (по дизайну
    агента), поэтому сообщения добавляются вручную и сохраняются
    ``_persist()`` — сетевые вызовы не выполняются.
    """

    def setUp(self) -> None:
        _TESTS_TMP.mkdir(parents=True, exist_ok=True)
        self.db_path = _TESTS_TMP / f"persist_{uuid.uuid4().hex}.db"

    def tearDown(self) -> None:
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)

    def _make_agent(self, repo: ProfileRepository) -> object:
        from src.agent import DeepSeekAgent
        storage = ContextStore(db_path=self.db_path, enabled=True)
        agent = DeepSeekAgent(
            api_key="k", token_counter=HeuristicCounter(),
            storage=storage, profile_repository=repo,
        )
        return agent

    def test_messages_saved_with_profile_id(self) -> None:
        repo = ProfileRepository(db_path=self.db_path, config=_config())
        agent = self._make_agent(repo)
        try:
            agent._append_message({"role": "user", "content": "вопрос"})
            agent._append_message({"role": "assistant", "content": "ответ"})
            agent._persist()
            default_id = repo.get_by_name(DEFAULT_PROFILE_NAME).id
            self.assertGreaterEqual(
                repo.count_profile_messages(default_id), 2
            )
        finally:
            agent.close()

    def test_cascade_delete_removes_history(self) -> None:
        """Тест 6: удаление профиля удаляет его сообщения из БД."""
        repo = ProfileRepository(db_path=self.db_path, config=_config())
        agent = self._make_agent(repo)
        try:
            agent._append_message({"role": "user", "content": "диалог A"})
            agent._persist()
            default_id = repo.get_by_name(DEFAULT_PROFILE_NAME).id
            doomed = repo.create(UserProfile(name="Одноразовый"))
            agent.set_profile(doomed.id)
            agent._append_message(
                {"role": "user", "content": "диалог одноразового профиля"}
            )
            agent._persist()
            self.assertGreaterEqual(
                repo.count_profile_messages(doomed.id), 1
            )
            self.assertTrue(repo.delete(doomed.id))
            self.assertEqual(repo.count_profile_messages(doomed.id), 0)
            # История других профилей не тронута
            self.assertGreaterEqual(
                repo.count_profile_messages(default_id), 1
            )
        finally:
            agent.close()

    def test_agent_reloads_history_for_profile_from_db(self) -> None:
        """set_profile: история профиля восстанавливается из БД."""
        repo = ProfileRepository(db_path=self.db_path, config=_config())
        agent = self._make_agent(repo)
        try:
            agent._append_message(
                {"role": "user", "content": "вопрос в БД"}
            )
            agent._persist()
            other = repo.create(UserProfile(name="B"))
            agent.set_profile(other.id)
            agent.close()

            # Новый агент: возвращаем Default активным (set_profile(B)
            # зафиксировал B в БД) → его история восстановится из messages.
            repo.set_active(repo.get_by_name(DEFAULT_PROFILE_NAME).id)
            agent2 = self._make_agent(repo)
            try:
                self.assertTrue(any(
                    "вопрос в БД" in m["content"]
                    for m in agent2.history
                ))
                agent2.set_profile(other.id)
                self.assertFalse(any(
                    "вопрос в БД" in m["content"]
                    for m in agent2.history
                ))
            finally:
                agent2.close()
        finally:
            agent.close()


# ---------------------------------------------------------------------------
# GUI: вкладки «Профиль» и «Настройки агента»
# ---------------------------------------------------------------------------

try:  # pragma: no cover — зависит от окружения
    import tkinter as tk
except ImportError:  # pragma: no cover
    tk = None  # type: ignore


class ProfileGuiTest(unittest.TestCase):
    """GUI персонализации: вкладки, форма профиля, настройки агента."""

    root: "tk.Tk"
    _ChatApplication = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = gui_helpers.get_root()
        if cls.root is None:
            raise unittest.SkipTest("графическая среда недоступна (tkinter/Tk)")
        from src.ui import ChatApplication
        cls._ChatApplication = ChatApplication

    def setUp(self) -> None:
        _TESTS_TMP.mkdir(parents=True, exist_ok=True)
        gui_helpers.reset_children(self.root)
        self.repo = _make_repo("gui")
        from src.agent import DeepSeekAgent
        self.agent = DeepSeekAgent(
            api_key="k", mock_mode=True, token_counter=HeuristicCounter(),
            profile_repository=self.repo,
        )
        self.app = self._ChatApplication(
            root=self.root, agent=self.agent,
        )
        self.root.update()

    def tearDown(self) -> None:
        self.agent.close()
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)

    def test_notebook_has_three_tabs(self) -> None:
        tabs = [self.app.notebook.tab(t, "text")
                for t in self.app.notebook.tabs()]
        self.assertIn("Чат", tabs)
        self.assertIn("Профиль", tabs)
        self.assertIn("Настройки агента", tabs)

    def test_profile_switcher_in_status_bar(self) -> None:
        self.assertEqual(self.app.profile_switch["values"], ("Default",))
        self.assertIn('Профиль "Default"',
                      self.app.profile_info_label.cget("text"))

    def test_create_profile_via_form(self) -> None:
        """Форма профиля: «Новый» → заполнение → «Сохранить»."""
        self.app._on_profile_new()
        self.app.profile_name_entry.insert(0, "Senior Dev")
        self.app.profile_display_entry.insert(0, "Антон")
        self.app.profile_style_entry.insert(0, "кратко, только код")
        self.app.profile_constraints_text.insert("1.0", "без эмодзи\nтолько Python")
        self.app.profile_context_text.insert("1.0", "role: разработчик")
        self.app._on_profile_save()
        self.root.update()

        saved = self.repo.get_by_name("Senior Dev")
        self.assertIsNotNone(saved)
        self.assertEqual(saved.constraints, ["без эмодзи", "только Python"])
        self.assertEqual(saved.context, {"role": "разработчик"})
        self.assertIn('"Senior Dev" сохранён',
                      self.app.status_label.cget("text"))
        # Список профилей обновился
        self.assertIn("Senior Dev", self.app.profile_combo["values"])

    def test_switch_profile_via_status_bar_reloads_chat(self) -> None:
        """Переключатель в статус-баре меняет активный профиль и историю."""
        self.agent.send_message("диалог первого профиля")
        self.app._refresh_chat()
        other = self.repo.create(UserProfile(name="B"))
        self.app._refresh_profile_tab()

        self.app.profile_switch_var.set("B")
        self.app._on_status_profile_switch()
        self.root.update()

        self.assertEqual(self.agent.profile.id, other.id)
        self.assertFalse(any(
            "первого профиля" in m["content"] for m in self.agent.history
        ))
        self.assertIn('Профиль "B"',
                      self.app.profile_info_label.cget("text"))

    def test_agent_settings_tab_saves_to_profile(self) -> None:
        """Вкладка «Настройки агента» сохраняет настройки активного профиля."""
        self.app.profile_model_var.set("deepseek-v4-pro")
        self.app.profile_temp_var.set(0.4)
        self.app.profile_tokens_var.set("1500")
        self.app.profile_prompt_text.delete("1.0", "end")
        self.app.profile_prompt_text.insert("1.0", "Особая базовая инструкция")
        self.app._on_profile_settings_save()
        self.root.update()

        settings = self.repo.get_settings(self.agent.profile.id)
        self.assertEqual(settings.model, "deepseek-v4-pro")
        self.assertEqual(settings.temperature, 0.4)
        self.assertEqual(settings.max_tokens, 1500)
        self.assertEqual(settings.base_system_prompt, "Особая базовая инструкция")
        # Применились к агенту «на лету»
        self.assertEqual(self.agent.temperature, 0.4)
        self.assertEqual(self.agent.max_tokens, 1500)

    def test_delete_profile_with_confirmation(self) -> None:
        """Удаление профиля: подтверждение, каскад, возврат на Default."""
        doomed = self.repo.create(UserProfile(name="Лишний"))
        self.app._refresh_profile_tab()
        self.app.profile_var.set("Лишний")
        self.app._on_profile_form_selected()

        with mock.patch("src.ui.messagebox.askyesno", return_value=True):
            self.app._on_profile_delete()
        self.root.update()

        self.assertIsNone(self.repo.get_by_id(doomed.id))
        self.assertEqual(self.agent.profile.name, DEFAULT_PROFILE_NAME)
        self.assertIn("удалён", self.app.status_label.cget("text"))


if __name__ == "__main__":
    unittest.main()
