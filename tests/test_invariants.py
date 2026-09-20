"""Тесты подсистемы инвариантов (инварианты, категории, пост-проверка).

Покрывают 14 тестов из ТЗ (офлайн, без сети):

1. Прямое нарушение — правила отказа и настойчивости в system-промпте.
2. Косвенное нарушение — тот же механизм (инвариант в промпте).
3. Нет нарушения — пост-проверка пропускает ответ без нарушения.
4. Настойчивость — правило «даже если пользователь настаивает» в промпте.
5. Удаление инварианта — промпт пересобирается без него.
6. Конфликт через LLM — валидатор находит противоречия (JSON).
7. Scope — профильный инвариант только для своего профиля, глобальный
   для всех.
8. Приоритет — «Безопасность» выше «Стиль кода» в блоке промпта.
9. Пост-проверка — валидатор ловит нарушение → перегенерация.
10. Лимит перегенераций — после 2 попыток предупреждение + unresolved.
11. UI — создание/редактирование/удаление инварианта и категории.
12. Системная категория — удалить нельзя.
13. Кнопка «Показать инварианты» открывает модалку.
14. Лог нарушений — отказ/перегенерация пишутся в invariant_violations.

Tk-тесты используют общий root из ``tests/gui_helpers.py`` и аккуратно
пропускаются без графической среды.

Запуск из корня проекта:
    python -m unittest tests.test_invariants -v
"""

from __future__ import annotations

import json
import pathlib
import shutil
import tempfile
import time
import unittest
from unittest import mock

from src.agent import DeepSeekAgent
from src.config import DEFAULTS, get_invariant_settings
from src.invariant_models import (
    SCOPE_GLOBAL,
    SCOPE_PROFILE,
    Invariant,
    InvariantCategory,
)
from src.invariant_prompt_builder import build_invariants_block
from src.invariant_repository import (
    PRESET_CATEGORIES,
    InvariantRepository,
)
from src.invariant_validator import (
    VALIDATOR_MARKER,
    Conflict,
    InvariantValidator,
    ValidationResult,
)
from src.models import UserProfile
from src.profile_repository import ProfileRepository
from src.storage import ContextStore
from src.tokens import HeuristicCounter

_TESTS_TMP = pathlib.Path(tempfile.gettempdir()) / "invariants_tests"


class InvariantsBase(unittest.TestCase):
    """Общая база: временная БД с профилями и инвариантами."""

    def setUp(self) -> None:
        _TESTS_TMP.mkdir(parents=True, exist_ok=True)
        self._tmp_dir = pathlib.Path(tempfile.mkdtemp(dir=_TESTS_TMP))
        self.db_path = self._tmp_dir / "invariants.db"
        self.profiles = ProfileRepository(db_path=self.db_path)
        self.repo = InvariantRepository(db_path=self.db_path)

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def make_category(self, name: str = "custom",
                      display: str = "Пользовательские",
                      priority: int = 100) -> InvariantCategory:
        """Создаёт категорию в БД (или возвращает существующую с таким именем)."""
        existing = next(
            (c for c in self.repo.get_all_categories() if c.name == name),
            None,
        )
        if existing is not None:
            return existing
        return self.repo.create_category(InvariantCategory(
            name=name, display_name=display, priority=priority,
        ))

    def make_invariant(self, category_id: int, title: str,
                       rule: str, scope: str = SCOPE_GLOBAL,
                       profile_id: int | None = None) -> Invariant:
        """Создаёт инвариант в БД и возвращает его."""
        return self.repo.create_invariant(Invariant(
            category_id=category_id, title=title, rule=rule,
            scope=scope, profile_id=profile_id,
        ))

    def make_agent(self, profile_repository: ProfileRepository | None = None,
                   invariant_repository: InvariantRepository | None = None,
                   validator: InvariantValidator | None = None) -> DeepSeekAgent:
        """Mock-агент с переданными подсистемами (без сети)."""
        return DeepSeekAgent(
            api_key="k", mock_mode=True,
            token_counter=HeuristicCounter(),
            profile_repository=profile_repository,
            invariant_repository=invariant_repository,
            invariant_validator=validator,
        )


# ---------------------------------------------------------------------------
# Миграция схемы и предустановленные данные
# ---------------------------------------------------------------------------

class MigrationTest(InvariantsBase):
    """Миграция: таблицы, предустановленные категории, примеры инвариантов."""

    def test_tables_and_indexes_created(self) -> None:
        import sqlite3

        conn = sqlite3.connect(str(self.db_path))
        try:
            names = {
                str(row[0]) for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            indexes = {
                str(row[0]) for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index'"
                )
            }
        finally:
            conn.close()
        for table in ("invariant_categories", "invariants",
                      "invariant_violations"):
            self.assertIn(table, names)
        for index in ("idx_invariants_category", "idx_invariants_scope"):
            self.assertIn(index, indexes)

    def test_preset_categories_added(self) -> None:
        categories = {c.name: c for c in self.repo.get_all_categories()}
        expected = {item[0]: item for item in PRESET_CATEGORIES}
        self.assertEqual(set(categories), set(expected))
        for name, (name_, display, _description, priority, is_system) \
                in expected.items():
            category = categories[name]
            self.assertEqual(category.display_name, display)
            self.assertEqual(category.priority, priority)
            self.assertEqual(category.is_system, bool(is_system))

    def test_sample_invariants_created_once(self) -> None:
        """При первой миграции создаются 1–2 инварианта-примера (глобальные)."""
        samples = self.repo.get_all_invariants()
        self.assertGreaterEqual(len(samples), 1)
        self.assertLessEqual(len(samples), 2)
        for invariant in samples:
            self.assertEqual(invariant.scope, SCOPE_GLOBAL)
            self.assertIsNone(invariant.profile_id)
        # Повторная миграция (новый репозиторий на той же БД) не дублирует
        InvariantRepository(db_path=self.db_path)
        self.assertEqual(len(self.repo.get_all_invariants()), len(samples))


# ---------------------------------------------------------------------------
# Категории: CRUD и системная защита
# ---------------------------------------------------------------------------

class CategoryCrudTest(InvariantsBase):
    """Категории: создание, обновление, удаление, системная защита."""

    def test_create_update_category(self) -> None:
        category = self.make_category("testing", "Тестирование", 70)
        stored = self.repo.get_category_by_id(category.id or 0)
        assert stored is not None
        self.assertEqual(stored.display_name, "Тестирование")
        stored.display_name = "Тесты"
        stored.priority = 65
        self.repo.update_category(stored)
        reloaded = self.repo.get_category_by_id(category.id or 0)
        assert reloaded is not None
        self.assertEqual(reloaded.display_name, "Тесты")
        self.assertEqual(reloaded.priority, 65)

    def test_delete_category_cascades_invariants(self) -> None:
        category = self.make_category("temp", "Временная", 90)
        invariant = self.make_invariant(category.id or 0, "Правило", "rule")
        self.assertTrue(self.repo.delete_category(category.id or 0))
        self.assertIsNone(self.repo.get_category_by_id(category.id or 0))
        self.assertIsNone(
            self.repo.get_invariant_by_id(invariant.id or 0)
        )

    def test_system_category_cannot_be_deleted(self) -> None:
        """Тест 12 ТЗ: категорию с is_system = 1 удалить нельзя."""
        security = next(
            c for c in self.repo.get_all_categories() if c.name == "security"
        )
        self.assertTrue(security.is_system)
        self.assertFalse(self.repo.delete_category(security.id or 0))
        self.assertIsNotNone(
            self.repo.get_category_by_id(security.id or 0)
        )

    def test_duplicate_category_name_rejected(self) -> None:
        self.repo.create_category(InvariantCategory(
            name="dup", display_name="Дубль",
        ))
        with self.assertRaises(ValueError):
            self.repo.create_category(InvariantCategory(
                name="dup", display_name="Дубль-2",
            ))


# ---------------------------------------------------------------------------
# Инварианты: CRUD и области действия (scope)
# ---------------------------------------------------------------------------

class InvariantCrudTest(InvariantsBase):
    """Инварианты: CRUD, валидация, счётчики."""

    def test_create_update_delete_invariant(self) -> None:
        category = self.make_category()
        invariant = self.make_invariant(
            category.id or 0, "Только Python", "Код только на Python 3.9+."
        )
        stored = self.repo.get_invariant_by_id(invariant.id or 0)
        assert stored is not None
        self.assertEqual(stored.rule, "Код только на Python 3.9+.")
        stored.title = "Только Python 3.10+"
        self.repo.update_invariant(stored)
        reloaded = self.repo.get_invariant_by_id(invariant.id or 0)
        assert reloaded is not None
        self.assertEqual(reloaded.title, "Только Python 3.10+")
        self.assertTrue(self.repo.delete_invariant(invariant.id or 0))
        self.assertIsNone(self.repo.get_invariant_by_id(invariant.id or 0))

    def test_invariant_validation(self) -> None:
        category = self.make_category()
        with self.assertRaises(ValueError):
            self.repo.create_invariant(Invariant(
                category_id=category.id or 0, title="", rule="r",
            ))
        with self.assertRaises(ValueError):
            self.repo.create_invariant(Invariant(
                category_id=category.id or 0, title="t", rule=" ",
            ))
        with self.assertRaises(ValueError):
            self.repo.create_invariant(Invariant(
                category_id=99999, title="t", rule="r",
            ))
        # Глобальный не может быть привязан к профилю, профильный — без профиля
        with self.assertRaises(ValueError):
            self.repo.create_invariant(Invariant(
                category_id=category.id or 0, title="t", rule="r",
                scope=SCOPE_GLOBAL, profile_id=1,
            ))
        with self.assertRaises(ValueError):
            self.repo.create_invariant(Invariant(
                category_id=category.id or 0, title="t", rule="r",
                scope=SCOPE_PROFILE, profile_id=None,
            ))

    def test_count_invariants(self) -> None:
        category = self.make_category()
        profile = self.profiles.create(UserProfile(name="Dev"))
        self.make_invariant(category.id or 0, "G1", "rule1",
                            scope=SCOPE_GLOBAL)
        self.make_invariant(category.id or 0, "P1", "rule2",
                            scope=SCOPE_PROFILE, profile_id=profile.id)
        counts = self.repo.count_invariants(profile.id)
        # + 2 глобальных инварианта-примера из миграции
        self.assertEqual(counts, {"total": 4, "global": 3, "profile": 1})


class ScopeTest(InvariantsBase):
    """Тест 7 ТЗ: области действия global/profile."""

    def test_profile_invariant_only_for_its_profile(self) -> None:
        category = self.make_category()
        dev = self.profiles.create(UserProfile(name="Dev"))
        other = self.profiles.create(UserProfile(name="Other"))
        self.make_invariant(category.id or 0, "Только Tkinter",
                            "UI только на Tkinter.",
                            scope=SCOPE_PROFILE, profile_id=dev.id)
        self.make_invariant(category.id or 0, "Глобальное", "Общее правило.",
                            scope=SCOPE_GLOBAL)

        dev_invariants = self.repo.get_invariants_for_profile(dev.id)
        other_invariants = self.repo.get_invariants_for_profile(other.id)
        dev_titles = {i.title for i in dev_invariants}
        other_titles = {i.title for i in other_invariants}
        # Профильный действует только для своего профиля...
        self.assertIn("Только Tkinter", dev_titles)
        self.assertNotIn("Только Tkinter", other_titles)
        # ...глобальный — для всех
        self.assertIn("Глобальное", dev_titles)
        self.assertIn("Глобальное", other_titles)
        # Без персонализации — только глобальные (2 примера из миграции +
        # созданный в тесте), профильных нет нигде
        no_profile = self.repo.get_invariants_for_profile(None)
        no_profile_titles = {i.title for i in no_profile}
        self.assertIn("Глобальное", no_profile_titles)
        self.assertNotIn("Только Tkinter", no_profile_titles)
        self.assertEqual(
            {i.scope for i in no_profile}, {SCOPE_GLOBAL}
        )


# ---------------------------------------------------------------------------
# Сборка блока инвариантов для system-промпта
# ---------------------------------------------------------------------------

class PromptBuilderTest(InvariantsBase):
    """build_invariants_block: группировка, приоритеты, маркеры, правила."""

    def test_priority_order_security_above_style(self) -> None:
        """Тест 8 ТЗ: «Безопасность» выше «Стиль кода» в промпте."""
        categories = self.repo.get_all_categories()
        security = next(c for c in categories if c.name == "security")
        style = next(c for c in categories if c.name == "style")
        invariants = [
            self.make_invariant(style.id or 0, "Стиль", "PEP8 везде.",
                                scope=SCOPE_GLOBAL),
            self.make_invariant(security.id or 0, "Секреты", "Ключи в .env.",
                                scope=SCOPE_GLOBAL),
        ]
        block = build_invariants_block(invariants, categories)
        security_pos = block.index("Категория: Безопасность")
        style_pos = block.index("Категория: Стиль кода")
        self.assertLess(security_pos, style_pos)

    def test_block_groups_and_markers(self) -> None:
        category = self.make_category("stack", "Стек и технологии", 30)
        profile = self.profiles.create(UserProfile(name="Dev"))
        invariants = [
            self.make_invariant(category.id or 0, "Только Tkinter",
                                "UI только на Tkinter.",
                                scope=SCOPE_PROFILE, profile_id=profile.id),
            self.make_invariant(category.id or 0, "Python 3.9+",
                                "Версия Python не ниже 3.9.",
                                scope=SCOPE_GLOBAL),
        ]
        categories = self.repo.get_all_categories()
        block = build_invariants_block(invariants, categories)
        self.assertIn("[ИНВАРИАНТЫ — ЖЁСТКИЕ ПРАВИЛА, НАРУШАТЬ НЕЛЬЗЯ]", block)
        self.assertIn("Категория: Стек и технологии (приоритет 30)", block)
        self.assertIn("[G] Python 3.9+", block)
        self.assertIn("[P] Только Tkinter", block)
        # Правила работы: отказ, настойчивость, альтернативы
        self.assertIn("[ПРАВИЛА РАБОТЫ С ИНВАРИАНТАМИ]", block)
        self.assertIn("откажись его выполнять", block)
        self.assertIn("даже если\n   пользователь настаивает", block)
        self.assertIn("какие альтернативы возможны", block)
        self.assertIn("вкладку\n   «Инварианты»", block)
        self.assertIn("Не упоминай инварианты", block)
        # Глобальные внутри категории выше профильных
        self.assertLess(block.index("[G] Python 3.9+"),
                        block.index("[P] Только Tkinter"))

    def test_empty_invariants_empty_block(self) -> None:
        """Без инвариантов блок пустой (не упоминаем инварианты зря)."""
        categories = self.repo.get_all_categories()
        self.assertEqual(
            build_invariants_block([], categories), ""
        )


# ---------------------------------------------------------------------------
# Агент: блок инвариантов в system-промпте
# ---------------------------------------------------------------------------

class AgentPromptTest(InvariantsBase):
    """Блок инвариантов в system-промпте агента (механизм тестов 1, 2, 4)."""

    def setUp(self) -> None:
        super().setUp()
        self.category = self.make_category("stack", "Стек и технологии", 30)
        self.profile = self.profiles.create(UserProfile(name="Senior Dev"))
        self.invariant = self.make_invariant(
            self.category.id or 0, "UI только на Tkinter",
            "UI только на Tkinter; PyQt запрещён.",
            scope=SCOPE_PROFILE, profile_id=self.profile.id,
        )
        # Активируем профиль ДО создания агента: промпт собирается
        # для активного профиля (global + его профильные инварианты)
        self.profiles.set_active(self.profile.id)
        self.agent = self.make_agent(
            profile_repository=self.profiles,
            invariant_repository=self.repo,
        )

    def test_invariants_block_in_system_prompt(self) -> None:
        """Инварианты попадают в промпт с правилами отказа (тесты 1, 2, 4)."""
        prompt = self.agent.system_prompt
        self.assertIn("UI только на Tkinter", prompt)
        self.assertIn("[ПРАВИЛА РАБОТЫ С ИНВАРИАНТАМИ]", prompt)
        self.assertIn("откажись его выполнять", prompt)
        self.assertIn("какие альтернативы возможны", prompt)
        self.assertIn("даже если", prompt)
        # Порядок: база → инварианты → профиль
        base_pos = prompt.index("[Базовая инструкция]")
        invariants_pos = prompt.index("[ИНВАРИАНТЫ")
        profile_pos = prompt.index("[Профиль пользователя")
        self.assertTrue(base_pos < invariants_pos < profile_pos)

    def test_other_profile_invariant_not_in_prompt(self) -> None:
        """Чужой профильный инвариант в промпт не попадает."""
        other = self.profiles.create(UserProfile(name="Other"))
        self.make_invariant(self.category.id or 0, "Чужое правило",
                            "Только для Other.", scope=SCOPE_PROFILE,
                            profile_id=other.id)
        prompt = self.agent.system_prompt
        self.assertNotIn("Чужое правило", prompt)

    def test_deleted_invariant_removed_from_prompt(self) -> None:
        """Тест 5 ТЗ: после удаления инварианта тот же запрос проходит."""
        self.assertIn("UI только на Tkinter", self.agent.system_prompt)
        self.assertTrue(self.repo.delete_invariant(self.invariant.id or 0))
        # Промпт пересобирается при следующем запросе: удалённого нет,
        # но глобальные инварианты-примеры остаются
        self.agent._refresh_profile_locked()
        prompt = self.agent.system_prompt
        self.assertNotIn("UI только на Tkinter", prompt)
        self.assertIn("[ИНВАРИАНТЫ", prompt)
        # Удаляем оставшиеся — блок исчезает из промпта полностью
        for invariant in self.repo.get_all_invariants():
            self.assertTrue(self.repo.delete_invariant(invariant.id or 0))
        self.agent._refresh_profile_locked()
        self.assertNotIn("[ИНВАРИАНТЫ", self.agent.system_prompt)

    def test_no_repository_no_block(self) -> None:
        """Без репозитория инвариантов поведение как раньше."""
        agent = self.make_agent()
        self.assertNotIn("[ИНВАРИАНТЫ", agent.system_prompt)


# ---------------------------------------------------------------------------
# Валидатор: разбор ответа модели
# ---------------------------------------------------------------------------

def _payload(content: str) -> dict:
    """Заглушка ответа chat.completions (для call_api валидатора)."""
    return {"content": content, "model": "m", "tokens": {}, "finish_reason": "stop"}


class ValidatorParseTest(InvariantsBase):
    """validate(): разбор JSON валидатора, отказы и ошибки."""

    def setUp(self) -> None:
        super().setUp()
        self.category = self.make_category()
        self.invariants = [
            self.make_invariant(self.category.id or 0, "Только Tkinter",
                                "UI только на Tkinter.", scope=SCOPE_GLOBAL),
        ]
        self.validator = InvariantValidator()

    def test_violation_detected(self) -> None:
        response = json.dumps({
            "violates": True, "refusal": False,
            "violated_invariant_id": self.invariants[0].id,
            "reason": "предлагает PyQt",
        }, ensure_ascii=False)
        result = self.validator.validate(
            response, self.invariants,
            call_api=lambda request: _payload(response),
        )
        self.assertTrue(result.violates)
        self.assertFalse(result.refusal)
        self.assertEqual(result.invariant_id, self.invariants[0].id)
        self.assertEqual(result.reason, "предлагает PyQt")

    def test_markdown_fences_and_text_tolerated(self) -> None:
        raw = (
            "Вот результат проверки:\n"
            "```json\n"
            + json.dumps({"violates": False, "refusal": True,
                          "violated_invariant_id": None, "reason": "отказ"},
                         ensure_ascii=False)
            + "\n```"
        )
        result = self.validator.validate(
            raw, self.invariants,
            call_api=lambda request: _payload(raw),
        )
        self.assertFalse(result.violates)
        self.assertTrue(result.refusal)

    def test_broken_json_is_not_violation(self) -> None:
        result = self.validator.validate(
            "не JSON вообще", self.invariants,
            call_api=lambda request: _payload("не JSON вообще"),
        )
        self.assertFalse(result.violates)
        self.assertIsNotNone(result.error)

    def test_api_error_is_not_violation(self) -> None:
        def broken(request) -> dict:
            raise RuntimeError("Сетевая ошибка: служебный запрос не прошёл.")

        result = self.validator.validate(
            "ответ", self.invariants, call_api=broken
        )
        self.assertFalse(result.violates)
        self.assertIn("Сетевая ошибка", result.error or "")

    def test_empty_invariants_skips_check(self) -> None:
        result = self.validator.validate(
            "ответ", [], call_api=lambda request: _payload("any")
        )
        self.assertFalse(result.violates)
        # Запрос к API не выполнялся: ошибок нет, проверки не было
        self.assertIsNone(result.error)

    def test_marker_in_service_request(self) -> None:
        """Служебный запрос помечен маркером (не путается с диалогом)."""
        seen: dict = {}

        def spy(request) -> dict:
            seen.update(request)
            return _payload('{"violates": false}')

        self.validator.validate("ответ", self.invariants, call_api=spy)
        self.assertIn(VALIDATOR_MARKER, seen["messages"][0]["content"])


# ---------------------------------------------------------------------------
# Валидатор: цикл перегенерации (тесты 9, 10 ТЗ)
# ---------------------------------------------------------------------------

class FakeRegenAgent:
    """Заглушка агента для validate_and_regenerate (метод перегенерации)."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, str]] = []

    def regenerate_last_response(self, user_query: str,
                                 violated_rule: str) -> str:
        self.calls.append((user_query, violated_rule))
        if not self.answers:
            raise RuntimeError("нет заготовленных ответов")
        return self.answers.pop(0)


class RegenerationTest(InvariantsBase):
    """validate_and_regenerate: перегенерация и лимит попыток."""

    def setUp(self) -> None:
        super().setUp()
        self.category = self.make_category()
        self.invariants = [
            self.make_invariant(self.category.id or 0, "Только Tkinter",
                                "UI только на Tkinter.", scope=SCOPE_GLOBAL),
        ]

    def _violates(self, invariant_id: int | None = None) -> str:
        return json.dumps({
            "violates": True, "refusal": False,
            "violated_invariant_id": invariant_id or self.invariants[0].id,
            "reason": "нарушение",
        }, ensure_ascii=False)

    def _ok(self) -> str:
        return json.dumps({
            "violates": False, "refusal": False,
            "violated_invariant_id": None, "reason": "",
        }, ensure_ascii=False)

    def test_violation_caught_and_regenerated(self) -> None:
        """Тест 9 ТЗ: нарушение поймано → перегенерация → нарушение снято."""
        validator = InvariantValidator(max_regenerations=2)
        responses = [
            _payload(self._violates()),   # первая проверка: нарушение
            _payload(self._ok()),         # проверка после перегенерации
        ]
        agent = FakeRegenAgent(["чистый ответ"])
        final, regens = validator.validate_and_regenerate(
            agent=agent, user_query="покажи PyQt",
            first_response="плохой ответ с PyQt",
            invariants=self.invariants,
            call_api=lambda request: responses.pop(0),
        )
        self.assertEqual(final, "чистый ответ")
        self.assertEqual(regens, 1)
        self.assertFalse(validator.last_unresolved)
        # Перегенерации передан запрос и формулировка нарушенного правила
        # (полный текст напоминания собирает агент — см. тест ниже)
        self.assertEqual(
            agent.calls,
            [("покажи PyQt", "UI только на Tkinter.")],
        )

    def test_regeneration_limit_reached(self) -> None:
        """Тест 10 ТЗ: после 2 попыток нарушение не устранено → флаг."""
        validator = InvariantValidator(max_regenerations=2)
        agent = FakeRegenAgent(["плохой ответ 2", "плохой ответ 3"])
        responses = [
            _payload(self._violates()),
            _payload(self._violates()),
            _payload(self._violates()),
        ]
        final, regens = validator.validate_and_regenerate(
            agent=agent, user_query="запрос",
            first_response="плохой ответ 1",
            invariants=self.invariants,
            call_api=lambda request: responses.pop(0),
        )
        self.assertEqual(regens, 2)  # лимит MAX_REGENERATIONS
        self.assertEqual(final, "плохой ответ 3")  # последний ответ
        self.assertTrue(validator.last_unresolved)
        self.assertEqual(len(responses), 0)  # третья проверка не нужна

    def test_refusal_is_not_violation(self) -> None:
        """Отказ по инварианту не считается нарушением (нет перегенерации)."""
        validator = InvariantValidator(max_regenerations=2)
        refusal = json.dumps({
            "violates": False, "refusal": True,
            "violated_invariant_id": self.invariants[0].id,
            "reason": "агент отказался и предложил альтернативы",
        }, ensure_ascii=False)
        agent = FakeRegenAgent([])
        final, regens = validator.validate_and_regenerate(
            agent=agent, user_query="сделай на PyQt",
            first_response="❌ Не могу: нарушается инвариант…",
            invariants=self.invariants,
            call_api=lambda request: _payload(refusal),
        )
        self.assertEqual(regens, 0)
        self.assertEqual(final, "❌ Не могу: нарушается инвариант…")
        self.assertEqual(agent.calls, [])


# ---------------------------------------------------------------------------
# Конфликты инвариантов через LLM (тест 6 ТЗ)
# ---------------------------------------------------------------------------

class ConflictTest(InvariantsBase):
    """find_conflicts: обнаружение противоречий между инвариантами."""

    def test_conflicts_detected(self) -> None:
        category = self.make_category("stack", "Стек и технологии", 30)
        existing = [
            self.make_invariant(category.id or 0, "Только Python",
                                "Весь код только на Python.",
                                scope=SCOPE_GLOBAL),
        ]
        new_invariant = Invariant(
            category_id=category.id or 0, title="Только Kotlin",
            rule="Весь код только на Kotlin.", scope=SCOPE_GLOBAL,
        )
        validator = InvariantValidator()
        raw = json.dumps({
            "conflicts": [
                {"id": existing[0].id,
                 "reason": "оба правила ограничивают язык программирования"}
            ]
        }, ensure_ascii=False)
        conflicts = validator.find_conflicts(
            new_invariant, existing,
            categories=self.repo.get_all_categories(),
            call_api=lambda request: _payload(raw),
        )
        self.assertEqual(len(conflicts), 1)
        assert isinstance(conflicts[0], Conflict)
        self.assertEqual(conflicts[0].invariant_id, existing[0].id)
        self.assertIn("язык", conflicts[0].reason)

    def test_no_conflicts(self) -> None:
        category = self.make_category()
        existing = [
            self.make_invariant(category.id or 0, "Правило", "rule",
                                scope=SCOPE_GLOBAL),
        ]
        validator = InvariantValidator()
        conflicts = validator.find_conflicts(
            Invariant(category_id=category.id or 0, title="Другое",
                      rule="другое правило"),
            existing,
            call_api=lambda request: _payload('{"conflicts": []}'),
        )
        self.assertEqual(conflicts, [])

    def test_api_error_gives_no_conflicts(self) -> None:
        """Ошибка LLM не блокирует сохранение (конфликтов «нет»)."""
        category = self.make_category()
        existing = [
            self.make_invariant(category.id or 0, "Правило", "rule",
                                scope=SCOPE_GLOBAL),
        ]
        validator = InvariantValidator()

        def broken(request) -> dict:
            raise RuntimeError("Сетевая ошибка.")

        self.assertEqual(
            validator.find_conflicts(
                Invariant(category_id=category.id or 0, title="N", rule="r"),
                existing, call_api=broken,
            ),
            [],
        )


# ---------------------------------------------------------------------------
# Интеграция с send_message: пост-проверка и лог нарушений
# ---------------------------------------------------------------------------

class StubValidator:
    """Управляемая заглушка валидатора (без LLM)."""

    def __init__(self, final_text: str | None, regenerations: int,
                 result: ValidationResult | None, enabled: bool = True,
                 min_response_length: int = 100,
                 max_regenerations: int = 2) -> None:
        self.enabled = enabled
        self.min_response_length = min_response_length
        self.max_regenerations = max_regenerations
        self.final_text = final_text
        self.regenerations = regenerations
        self.last_result = result
        self.last_unresolved = bool(result and result.violates)
        self.calls: list[dict] = []

    def validate_and_regenerate(self, agent, user_query, first_response,
                                invariants, call_api=None):
        self.calls.append({
            "user_query": user_query,
            "first_response": first_response,
            "invariants": list(invariants),
        })
        return (self.final_text
                if self.final_text is not None else first_response,
                self.regenerations)


LONG_MOCK_ANSWER = (
    "Это ответ в mock-режиме: реальный запрос к DeepSeek API не отправлялся, "
    "поэтому здесь размещён длинный текст, превышающий сто символов."
)


class AgentPostCheckTest(InvariantsBase):
    """send_message: пост-проверка, замена истории, лог нарушений."""

    def setUp(self) -> None:
        super().setUp()
        self.category = self.make_category()
        self.profile = self.profiles.create(UserProfile(name="Dev"))
        self.invariant = self.make_invariant(
            self.category.id or 0, "Только Tkinter",
            "UI только на Tkinter.", scope=SCOPE_PROFILE,
            profile_id=self.profile.id,
        )
        # Активируем профиль: пост-проверка берёт global + его инварианты
        self.profiles.set_active(self.profile.id)

    def _agent_with_stub(self, stub: StubValidator) -> DeepSeekAgent:
        agent = self.make_agent(
            profile_repository=self.profiles,
            invariant_repository=self.repo,
            validator=stub,
        )
        return agent

    def _send_long(self, agent: DeepSeekAgent, text: str) -> dict:
        """send_message с детерминированно длинным mock-ответом.

        Стандартный mock-ответ выбирается случайно и может быть короче
        MIN_RESPONSE_LENGTH — фиксируем длинный, чтобы условие запуска
        пост-проверки выполнялось всегда.
        """
        payload = {"content": LONG_MOCK_ANSWER, "model": "m",
                   "tokens": {}, "finish_reason": "stop"}
        with mock.patch.object(agent, "_mock_response", return_value=payload):
            return agent.send_message(text)

    def test_regenerated_text_replaces_history_and_response(self) -> None:
        """Тест 9 ТЗ (интеграция): финальный текст заменяет ответ в истории."""
        stub = StubValidator(
            final_text="Исправленный ответ на Tkinter, достаточно длинный "
                       "для пост-проверки — больше ста символов, чтобы "
                       "проверка точно запустилась.",
            regenerations=1,
            result=ValidationResult(violates=False),
        )
        agent = self._agent_with_stub(stub)
        result = self._send_long(agent, "Покажи решение на PyQt")
        self.assertIsNone(result["error"])
        self.assertEqual(result["regenerations"], 1)
        self.assertFalse(result["invariant_warning"])
        self.assertTrue(str(result["text"]).startswith("Исправленный ответ"))
        # В истории остался финальный текст, а не исходный mock-ответ
        self.assertEqual(agent.history[-1]["content"], result["text"])
        # Проверка получила активные инварианты (global + профильные)
        self.assertEqual(len(stub.calls), 1)
        ids = [i.id for i in stub.calls[0]["invariants"]]
        self.assertIn(self.invariant.id, ids)

    def test_limit_reached_sets_warning_and_logs(self) -> None:
        """Тест 10 + 14 ТЗ: лимит исчерпан → warning + запись unresolved."""
        stub = StubValidator(
            final_text=None,  # текст не изменился
            regenerations=2,
            result=ValidationResult(violates=True, invariant_id=self.invariant.id,
                                    reason="нарушение осталось"),
        )
        agent = self._agent_with_stub(stub)
        result = self._send_long(agent, "Опять PyQt")
        self.assertTrue(result["invariant_warning"])
        self.assertEqual(result["regenerations"], 2)
        violations = self.repo.get_violations()
        self.assertEqual(len(violations), 1)
        self.assertTrue(violations[0].regenerated)
        self.assertFalse(violations[0].resolved)
        self.assertEqual(violations[0].invariant_id, self.invariant.id)
        self.assertEqual(violations[0].profile_id, self.profile.id)

    def test_refusal_logged_as_resolved(self) -> None:
        """Тест 14 ТЗ: отказ агента по инварианту попадает в лог (resolved)."""
        stub = StubValidator(
            final_text=None,
            regenerations=0,
            result=ValidationResult(refusal=True,
                                    invariant_id=self.invariant.id,
                                    reason="агент отказался"),
        )
        agent = self._agent_with_stub(stub)
        self._send_long(agent, "Сделай на PyQt")
        violations = self.repo.get_violations()
        self.assertEqual(len(violations), 1)
        self.assertFalse(violations[0].regenerated)
        self.assertTrue(violations[0].resolved)

    def test_short_response_not_checked(self) -> None:
        """Короткие ответы (< MIN_RESPONSE_LENGTH) не проверяются."""
        stub = StubValidator(final_text=None, regenerations=0,
                             result=ValidationResult(),
                             min_response_length=100)
        agent = self._agent_with_stub(stub)
        with mock.patch.object(agent, "_mock_response", return_value={
            "content": "коротко", "model": "m", "tokens": {},
            "finish_reason": "stop",
        }):
            result = agent.send_message("привет")
        self.assertEqual(stub.calls, [])
        self.assertNotIn("regenerations", result)
        self.assertNotIn("invariant_warning", result)

    def test_disabled_validator_skips_check(self) -> None:
        """ENABLE_POST_CHECK=false — проверка не выполняется."""
        stub = StubValidator(final_text=None, regenerations=0,
                             result=ValidationResult(), enabled=False)
        agent = self._agent_with_stub(stub)
        result = agent.send_message("вопрос обычный")
        self.assertEqual(stub.calls, [])
        self.assertNotIn("invariant_warning", result)

    def test_no_invariants_skips_check(self) -> None:
        """Нет активных инвариантов — проверка не запускается."""
        for invariant in self.repo.get_all_invariants():
            self.assertTrue(self.repo.delete_invariant(invariant.id or 0))
        stub = StubValidator(final_text=None, regenerations=0,
                             result=ValidationResult())
        agent = self._agent_with_stub(stub)
        self._send_long(agent, "вопрос обычный")
        self.assertEqual(stub.calls, [])

    def test_real_validator_with_stubbed_api(self) -> None:
        """Полный цикл с настоящим валидатором: API и перегенерация — заглушки."""
        validator = InvariantValidator(max_regenerations=2)
        agent = self._agent_with_stub(validator)  # type: ignore[arg-type]
        violates = json.dumps({
            "violates": True, "refusal": False,
            "violated_invariant_id": self.invariant.id, "reason": "PyQt",
        }, ensure_ascii=False)
        ok = json.dumps({
            "violates": False, "refusal": False,
            "violated_invariant_id": None, "reason": "",
        }, ensure_ascii=False)
        responses = [
            {"content": violates, "model": "m", "tokens": {},
             "finish_reason": "stop"},
            {"content": ok, "model": "m", "tokens": {},
             "finish_reason": "stop"},
        ]
        with mock.patch.object(
            agent, "_invariant_call_api",
            side_effect=lambda request: responses.pop(0),
        ), mock.patch.object(
            agent, "regenerate_last_response",
            return_value="Чистый ответ без нарушений, достаточно длинный "
                         "для пост-проверки — больше ста символов точно.",
        ) as regen:
            result = self._send_long(agent, "Покажи PyQt")
        self.assertEqual(result["regenerations"], 1)
        self.assertFalse(result["invariant_warning"])
        self.assertTrue(str(result["text"]).startswith("Чистый ответ"))
        regen.assert_called_once()
        self.assertFalse(validator.last_unresolved)
        # Нарушение с перегенерацией зафиксировано как устранённое
        violations = self.repo.get_violations()
        self.assertEqual(len(violations), 1)
        self.assertTrue(violations[0].regenerated)
        self.assertTrue(violations[0].resolved)

    def test_mock_mode_without_callables_degrades_safely(self) -> None:
        """Mock-режим: служебный вызов невозможен — пост-проверка молчит."""
        validator = InvariantValidator(max_regenerations=2)
        agent = self._agent_with_stub(validator)  # type: ignore[arg-type]
        result = agent.send_message("вопрос")
        self.assertIsNone(result["error"])
        self.assertEqual(result.get("regenerations", 0), 0)

    def test_regenerate_last_response_builds_reminder(self) -> None:
        """Перегенерация: нарушенный ответ убран, добавлено напоминание."""
        agent = self._agent_with_stub(
            StubValidator(final_text=None, regenerations=0,
                          result=ValidationResult())
        )
        seen: dict = {}

        def spy_call(request) -> dict:
            seen.update(request)
            return {"content": "новый ответ", "model": "m", "tokens": {},
                    "finish_reason": "stop"}

        with mock.patch.object(agent, "_call_service_api",
                               side_effect=spy_call):
            text = agent.regenerate_last_response("покажи PyQt",
                                                  "UI только на Tkinter.")
        self.assertEqual(text, "новый ответ")
        messages = seen["messages"]
        # Последнее сообщение — user-напоминание, а не нарушивший ответ
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("Предыдущий ответ нарушил инвариант: "
                      "UI только на Tkinter.", messages[-1]["content"])
        self.assertIn("Сгенерируй ответ заново", messages[-1]["content"])
        self.assertIn("Пользовательский запрос: покажи PyQt",
                      messages[-1]["content"])
        # Нарушивший ответ ассистента в запрос не попал
        assistant_texts = [m["content"] for m in messages
                           if m["role"] == "assistant"]
        self.assertNotIn(LONG_MOCK_ANSWER, assistant_texts)


# ---------------------------------------------------------------------------
# Конфигурация пост-проверки
# ---------------------------------------------------------------------------

class ConfigTest(unittest.TestCase):
    """get_invariant_settings: значения по умолчанию и валидация."""

    def test_defaults(self) -> None:
        settings = get_invariant_settings(dict(DEFAULTS))
        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["max_regenerations"], 2)
        self.assertEqual(settings["min_response_length"], 100)

    def test_env_overrides_and_garbage(self) -> None:
        config = dict(DEFAULTS)
        config["ENABLE_POST_CHECK"] = False
        config["MAX_REGENERATIONS"] = "5"
        config["MIN_RESPONSE_LENGTH"] = "abc"  # мусор → значение по умолчанию
        settings = get_invariant_settings(config)
        self.assertFalse(settings["enabled"])
        self.assertEqual(settings["max_regenerations"], 5)
        self.assertEqual(settings["min_response_length"], 100)

    def test_range_validation(self) -> None:
        config = dict(DEFAULTS)
        config["MAX_REGENERATIONS"] = "99"  # вне [0, 10] → 2
        config["MIN_RESPONSE_LENGTH"] = "-5"  # вне [0, 100000] → 100
        settings = get_invariant_settings(config)
        self.assertEqual(settings["max_regenerations"], 2)
        self.assertEqual(settings["min_response_length"], 100)

    def test_validator_from_settings(self) -> None:
        settings = get_invariant_settings(dict(DEFAULTS))
        validator = InvariantValidator(**settings)
        self.assertTrue(validator.enabled)
        self.assertEqual(validator.max_regenerations, 2)
        self.assertEqual(validator.min_response_length, 100)


# ---------------------------------------------------------------------------
# GUI: вкладка «Инварианты», модалка панели задачи, статус-бар
# ---------------------------------------------------------------------------

try:  # pragma: no cover — зависит от окружения
    import tkinter as tk
except ImportError:  # pragma: no cover
    tk = None  # type: ignore

from tests import gui_helpers  # noqa: E402


class InvariantsGuiBase(unittest.TestCase):
    """Общий root и пересоздание виджетов приложения (как в test_memory_ui)."""

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
        self._tmp_dir = pathlib.Path(tempfile.mkdtemp(dir=_TESTS_TMP))
        self.db_path = self._tmp_dir / "gui_invariants.db"
        self.profiles = ProfileRepository(db_path=self.db_path)
        self.repo = InvariantRepository(db_path=self.db_path)

        previous = getattr(self, "app", None)
        if previous is not None:
            for closer in ("_close_invariant_conflict_dialog",
                           "_close_invariant_dialog",
                           "_close_category_dialog",
                           "_close_invariants_window"):
                try:
                    getattr(previous, closer)()
                except Exception:
                    pass
        gui_helpers.reset_children(self.root)
        self.agent = DeepSeekAgent(
            api_key="k", mock_mode=True,
            token_counter=HeuristicCounter(),
            storage=ContextStore(db_path=self.db_path, enabled=True),
            profile_repository=self.profiles,
            invariant_repository=self.repo,
        )
        self.app = self._ChatApplication(root=self.root, agent=self.agent)
        self.root.update()

    def tearDown(self) -> None:
        for closer in ("_close_invariant_conflict_dialog",
                       "_close_invariant_dialog",
                       "_close_category_dialog",
                       "_close_invariants_window"):
            try:
                getattr(self.app, closer)()
            except Exception:
                pass
        shutil.rmtree(self._tmp_dir, ignore_errors=True)


class InvariantsTabGuiTest(InvariantsGuiBase):
    """Вкладка «Инварианты»: панели, создание/редактирование/удаление."""

    def setUp(self) -> None:
        super().setUp()
        self.category = self.repo.create_category(InvariantCategory(
            name="stack_gui", display_name="Стек GUI", priority=35,
        ))
        self.profile = self.profiles.create(UserProfile(name="Dev"))
        self.invariant = self.repo.create_invariant(Invariant(
            category_id=self.category.id or 0, title="Только Tkinter",
            description="Никаких PyQt.", rule="UI только на Tkinter.",
            scope=SCOPE_PROFILE, profile_id=self.profile.id,
        ))
        self.app._refresh_invariants_tab()
        self._show_profile("Dev")
        self.root.update()

    def _show_profile(self, name: str) -> None:
        """Переключает вкладку «Инварианты» на профиль ``name``."""
        if name in self.app._invariants_profiles:
            self.app.invariants_profile_var.set(name)
            self.app._refresh_invariants_list()
        self.root.update()

    def _select_category(self, category_id: int) -> None:
        """Выделяет категорию с ``category_id`` в левом списке."""
        categories = self.app._invariant_categories
        index = next(
            i for i, category in enumerate(categories)
            if category.id == category_id
        )
        self.app.invariants_category_list.selection_clear(0, "end")
        self.app.invariants_category_list.selection_set(index)
        self.app._on_invariants_category_select()
        self.root.update()

    def _select_invariant(self) -> None:
        """Открывает категорию с «Только Tkinter» и выделяет его в списке."""
        self._select_category(self.category.id)
        for index in range(self.app.invariants_list.size()):
            if "Только Tkinter" in self.app.invariants_list.get(index):
                self.app.invariants_list.selection_clear(0, "end")
                self.app.invariants_list.selection_set(index)
                return
        self.fail("Инвариант «Только Tkinter» не найден в списке")

    def test_tab_present_with_categories(self) -> None:
        """Вкладка «Инварианты» есть, категории загружены (по приоритету)."""
        tabs = [self.app.notebook.tab(i, "text")
                for i in range(self.app.notebook.index("end"))]
        self.assertIn("Инварианты", tabs)
        self.assertEqual(tabs.index("Инварианты"), tabs.index("Профиль") + 1)
        self.assertGreaterEqual(
            self.app.invariants_category_list.size(), 8
        )  # 7 системных + созданная в тесте
        first = self.app.invariants_category_list.get(0)
        self.assertIn("Безопасность", first)

    def test_category_filter_updates_invariants_list(self) -> None:
        """Выбор категории фильтрует список инвариантов."""
        categories = self.app._invariant_categories
        stack_index = next(
            index for index, category in enumerate(categories)
            if category.id == self.category.id
        )
        self.app.invariants_category_list.selection_clear(0, "end")
        self.app.invariants_category_list.selection_set(stack_index)
        self.app._on_invariants_category_select()
        self.root.update()
        rows = list(self.app.invariants_list.get(0, "end"))
        self.assertTrue(any("Только Tkinter" in row for row in rows))
        self.assertTrue(any("[P]" in row for row in rows))

    def test_create_invariant_via_dialog(self) -> None:
        """Создание инварианта через форму (с LLM-проверкой конфликтов)."""
        with mock.patch.object(
            self.app.agent, "check_invariant_conflicts", return_value=[],
        ):
            self.app._on_new_invariant()
            dialog = self.app._invariant_dialog
            self.assertIsNotNone(dialog)
            # Целевой профиль по умолчанию — выбранный на вкладке
            dialog.invariant_category_var.set("Стек GUI")
            dialog.invariant_title_entry.insert(0, "Новый инвариант")
            dialog.invariant_rule_text.insert("1.0", "Новое правило для LLM.")
            self.app._on_invariant_dialog_save()
            # Результат «фоновой» проверки подставляем в главном потоке
            self.app._on_invariant_conflict_check_done([])
            self.root.update()
        titles = [i.title for i in self.repo.get_all_invariants()]
        self.assertIn("Новый инвариант", titles)
        self.assertIsNone(self.app._invariant_dialog)

    def test_conflict_warning_does_not_block_save(self) -> None:
        """Тест 6 ТЗ (UI): конфликт предупреждает, но не блокирует."""
        with mock.patch.object(
            self.app.agent, "check_invariant_conflicts",
            return_value=[Conflict(
                invariant_id=self.invariant.id, reason="оба про UI",
            )],
        ):
            self.app._on_new_invariant()
            dialog = self.app._invariant_dialog
            dialog.invariant_category_var.set("Стек GUI")
            dialog.invariant_title_entry.insert(0, "Конфликтное правило")
            dialog.invariant_rule_text.insert("1.0", "Ещё правило про UI.")
            self.app._on_invariant_dialog_save()
            self.app._on_invariant_conflict_check_done([Conflict(
                invariant_id=self.invariant.id, reason="оба про UI",
            )])
            self.root.update()
            # Появилось предупреждение с кнопками «Всё равно сохранить»
            self.assertIsNotNone(self.app._invariant_conflict_dialog)
            self.app._on_invariant_conflict_save_anyway()
            self.root.update()
        titles = [i.title for i in self.repo.get_all_invariants()]
        self.assertIn("Конфликтное правило", titles)

    def test_conflict_cancel_does_not_save(self) -> None:
        with mock.patch.object(
            self.app.agent, "check_invariant_conflicts",
            return_value=[Conflict(invariant_id=self.invariant.id,
                                   reason="конфликт")],
        ):
            self.app._on_new_invariant()
            dialog = self.app._invariant_dialog
            dialog.invariant_category_var.set("Стек GUI")
            dialog.invariant_title_entry.insert(0, "Отменённое правило")
            dialog.invariant_rule_text.insert("1.0", "Правило.")
            self.app._on_invariant_dialog_save()
            self.app._on_invariant_conflict_check_done([Conflict(
                invariant_id=self.invariant.id, reason="конфликт",
            )])
            self.app._on_invariant_conflict_cancel()
            self.root.update()
        titles = [i.title for i in self.repo.get_all_invariants()]
        self.assertNotIn("Отменённое правило", titles)

    def test_edit_invariant_via_dialog(self) -> None:
        self._select_invariant()
        self.app._on_edit_invariant()
        dialog = self.app._invariant_dialog
        self.assertIsNotNone(dialog)
        self.assertEqual(dialog.invariant_title_entry.get(), "Только Tkinter")
        dialog.invariant_title_entry.delete(0, "end")
        dialog.invariant_title_entry.insert(0, "Только ttk и tk")
        with mock.patch.object(
            self.app.agent, "check_invariant_conflicts", return_value=[],
        ):
            self.app._on_invariant_dialog_save()
            self.app._on_invariant_conflict_check_done([])
            self.root.update()
        stored = self.repo.get_invariant_by_id(self.invariant.id or 0)
        assert stored is not None
        self.assertEqual(stored.title, "Только ttk и tk")

    def test_delete_invariant_via_ui(self) -> None:
        self._select_invariant()
        with mock.patch(
            "src.ui.messagebox.askyesno", return_value=True,
        ):
            self.app._on_delete_invariant()
        self.root.update()
        self.assertIsNone(
            self.repo.get_invariant_by_id(self.invariant.id or 0)
        )

    def test_category_create_and_system_delete_rejected(self) -> None:
        """Создание категории и запрет удаления системной (тест 12 ТЗ)."""
        self.app._on_new_category()
        dialog = self.app._invariant_dialog
        self.assertIsNone(dialog)  # открылась форма категории
        category_dialog = self.app._category_dialog
        self.assertIsNotNone(category_dialog)
        category_dialog.category_name_entry.insert(0, "review")
        category_dialog.category_display_entry.insert(0, "Ревью")
        category_dialog.category_priority_var.set("45")
        self.app._on_category_dialog_save()
        self.root.update()
        stored = next((c for c in self.repo.get_all_categories()
                       if c.name == "review"), None)
        self.assertIsNotNone(stored)
        self.assertEqual(stored.display_name, "Ревью")

        # Системную категорию удалить нельзя
        security = next(c for c in self.repo.get_all_categories()
                        if c.name == "security")
        index = self.app._invariant_categories.index(security)
        self.app.invariants_category_list.selection_clear(0, "end")
        self.app.invariants_category_list.selection_set(index)
        self.app._on_delete_category()
        self.root.update()
        self.assertIsNotNone(
            self.repo.get_category_by_id(security.id or 0)
        )


class InvariantsModalGuiTest(InvariantsGuiBase):
    """Модалка «Показать инварианты» и статус-бар (тесты 13 + статус)."""

    def setUp(self) -> None:
        super().setUp()
        self.category = self.repo.create_category(InvariantCategory(
            name="stack_modal", display_name="Стек Модал", priority=35,
        ))
        self.profile = self.profiles.create(UserProfile(name="Dev"))
        self.repo.create_invariant(Invariant(
            category_id=self.category.id or 0, title="Только Tkinter",
            rule="UI только на Tkinter.",
            scope=SCOPE_PROFILE, profile_id=self.profile.id,
        ))

    def test_task_panel_button_exists(self) -> None:
        """Кнопка «📋 Инварианты» есть в панели задачи рядом с остальными."""
        self.assertTrue(str(self.app.task_invariants_btn.cget("text"))
                        .find("Инварианты") >= 0)

    def test_modal_opens_and_switches_to_tab(self) -> None:
        """Тест 13 ТЗ: модалка открывается, «Перейти к редактированию»
        переключает на вкладку «Инварианты»."""
        self.app._on_show_invariants()
        self.root.update()
        window = self.app._invariants_window
        self.assertIsNotNone(window)
        # Содержимое: активный профиль + глобальные, маркеры [G]/[P]
        # (проверяем через закрытие — текст в Text-виджете окна)
        self.app._on_invariants_goto_edit()
        self.root.update()
        self.assertIsNone(self.app._invariants_window)
        current = self.app.notebook.tab(self.app.notebook.select(), "text")
        self.assertEqual(current, "Инварианты")

    def test_status_bar_counts(self) -> None:
        """Статус-бар: Инвариантов: N (глобальных: X, профильных: Y)."""
        self.agent.set_profile(self.profile.id)
        self.app._update_invariant_status()
        text = str(self.app.invariant_info_label.cget("text"))
        counts = self.repo.count_invariants(self.profile.id)
        self.assertIn(f"Инвариантов: {counts['total']}", text)
        self.assertIn(f"глобальных: {counts['global']}", text)
        self.assertIn(f"профильных: {counts['profile']}", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
