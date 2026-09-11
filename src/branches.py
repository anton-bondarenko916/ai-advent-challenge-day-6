"""Ветки диалога: checkpoint, развилки и независимое продолжение.

Модуль реализует стратегию **Branching** — работу с несколькими ветками
одного диалога:

* **checkpoint** — «сохраниться здесь»: создаётся ветка-копия текущего диалога
  (``create(from_index=None)``);
* **развилка** — ветка от любого выбранного сообщения
  (``create(from_index=i)``): копируется префикс диалога до этого места;
* **независимое продолжение** — сообщения веток не смешиваются;
* **переключение** — ``switch()`` меняет активную ветку; у каждой ветки свои
  сообщения, свои артефакты (facts) и своя подстратегия контекста.

История копируется **физически** (без «общего предка»): история в проекте
ограничена, а копия префикса проще и надёжнее в сопровождении.

Ветки живут в SQLite (таблица ``branches`` + ``messages.branch_id``); если
хранилище отключено, менеджер работает в памяти — развилки доступны в рамках
сеанса. Ошибки БД не пробрасываются наружу (лог внутри хранилища), а
некорректные операции (удаление последней ветки, лимит веток, ветка с детьми
без ``cascade``) поднимают :class:`~src.utils.AgentConfigError` с понятным
текстом.

Модуль не импортирует tkinter.
"""

from __future__ import annotations

import copy
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from . import utils
from .facts import FactsStore
from .storage import ROOT_BRANCH_ID, ROOT_BRANCH_NAME, ContextStore
from .utils import AgentConfigError

logger = utils.get_logger("deepseek-chat.branches")

#: Тип сообщения чата
Message = Dict[str, str]

#: Максимум веток по умолчанию
DEFAULT_BRANCH_MAX = 20

#: Шаблон автоимени ветки
_AUTO_NAME_RE = re.compile(r"^ветка-(\d+)$")


def _now() -> str:
    """Текущее локальное время в формате ISO (секунды)."""
    return datetime.now().isoformat(timespec="seconds")


def _split_system(messages: Sequence[Message]) -> tuple:
    """Делит историю на системный промпт и сообщения диалога.

    Args:
        messages: история вида ``[system, user, assistant, ...]``.

    Returns:
        Кортеж (system_messages, dialog_messages).
    """
    items = list(messages or [])
    if items and items[0].get("role") == "system":
        return [dict(items[0])], [dict(m) for m in items[1:]]
    return [], [dict(m) for m in items]


def copy_artifacts(artifacts: Optional[Dict[str, Any]],
                   prefix_length: int) -> Dict[str, Any]:
    """Копирует артефакты диалога для новой ветки.

    Факты копируются полностью; конспект (если он вообще используется) —
    только если он не выходит за границу скопированного префикса.

    Args:
        artifacts: исходные артефакты ({"facts", "summary", "counter"}).
        prefix_length: сколько сообщений диалога скопировано в ветку.

    Returns:
        Новый словарь артефактов.
    """
    source = artifacts or {}
    result: Dict[str, Any] = {}

    facts = source.get("facts")
    if isinstance(facts, FactsStore):
        clone = FactsStore(
            categories=facts.categories,
            max_items=facts.max_items,
            max_tokens=facts.max_tokens,
            counter=facts.counter,
        )
        clone.from_dict(facts.as_dict())
        result["facts"] = clone
    elif isinstance(facts, dict):
        result["facts"] = copy.deepcopy(facts)

    summary = source.get("summary")
    if isinstance(summary, dict) and str(summary.get("text") or "").strip():
        try:
            covered = int(summary.get("covered") or 0)
        except (TypeError, ValueError):
            covered = 0
        result["summary"] = copy.deepcopy(summary) if covered <= prefix_length else None
    else:
        result["summary"] = None

    if "counter" in source:
        result["counter"] = source["counter"]
    return result


def facts_to_plain(facts: Any) -> Dict[str, Any]:
    """Приводит факты к сериализуемому виду (для хранения и копий)."""
    if isinstance(facts, FactsStore):
        return facts.as_dict()
    if isinstance(facts, dict):
        return copy.deepcopy(facts)
    return {}


class BranchManager:
    """Дерево веток диалога поверх :class:`~src.storage.ContextStore`.

    Attributes:
        default_strategy: подстратегия контекста для новых веток.
        max_branches: максимум веток (``BRANCH_MAX``).

    Приватные атрибуты:
        _branches: ветки по id (id → словарь ветки).
        _active: id активной ветки.
        _memory_messages: сообщения веток при работе без БД.
        _memory_facts: факты веток при работе без БД.
    """

    def __init__(self, store: Optional[ContextStore] = None,
                 max_branches: int = DEFAULT_BRANCH_MAX,
                 default_strategy: str = "sliding_window") -> None:
        self._store = store
        self.max_branches = max(1, int(max_branches))
        self.default_strategy = str(default_strategy or "sliding_window")
        self._branches: Dict[int, Dict[str, Any]] = {}
        self._active: int = ROOT_BRANCH_ID
        self._memory_messages: Dict[int, List[Message]] = {}
        self._memory_facts: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Состояние
    # ------------------------------------------------------------------

    @property
    def persistence(self) -> bool:
        """True, если ветки сохраняются в БД."""
        return self._store is not None and bool(getattr(self._store, "enabled", False))

    def load(self) -> None:
        """Читает ветки и активную ветку из хранилища.

        При отсутствии веток создаётся корневая ``main`` (в БД — на уровне
        хранилища, в памяти — виртуальная).
        """
        rows: List[Dict[str, Any]] = []
        if self._store is not None:
            try:
                rows = self._store.load_branches()
            except Exception as exc:  # хранилище не должно ронять запуск
                logger.warning("Не удалось прочитать ветки: %s", exc)
                rows = []

        if rows:
            self._branches = {int(row["id"]): dict(row) for row in rows}
            active: Optional[int] = None
            if self._store is not None:
                active = self._store.get_active_branch()
            self._active = int(active) if active in self._branches \
                else min(self._branches)
        else:
            self._branches = {}
            self._active = ROOT_BRANCH_ID

        if ROOT_BRANCH_ID not in self._branches:
            self._branches[ROOT_BRANCH_ID] = self._virtual_branch(
                ROOT_BRANCH_ID, ROOT_BRANCH_NAME, None
            )
            self._active = ROOT_BRANCH_ID
        logger.debug(
            "Ветки загружены: %d шт., активная — %s.",
            len(self._branches), self.get_active().get("name"),
        )

    def _virtual_branch(self, branch_id: int, name: str,
                        parent_id: Optional[int]) -> Dict[str, Any]:
        """Ветка «в памяти» (когда хранилище отключено)."""
        now = _now()
        return {
            "id": int(branch_id),
            "name": str(name),
            "parent_id": parent_id,
            "forked_at": 0,
            "strategy": self.default_strategy,
            "created_at": now,
            "updated_at": now,
            "messages": 0,
        }

    def active_id(self) -> int:
        """Идентификатор активной ветки."""
        return int(self._active)

    def get(self, branch_id: int) -> Optional[Dict[str, Any]]:
        """Ветка по идентификатору (None, если её нет)."""
        try:
            return self._branches.get(int(branch_id))
        except (TypeError, ValueError):
            return None

    def get_active(self) -> Dict[str, Any]:
        """Активная ветка (всегда существует)."""
        branch = self._branches.get(self._active)
        if branch is None:  # страховка: состояние всегда согласовано
            name = ROOT_BRANCH_NAME if self._active == ROOT_BRANCH_ID \
                else f"ветка-{self._active}"
            branch = self._virtual_branch(self._active, name, None)
            self._branches[self._active] = branch
        return branch

    def count(self) -> int:
        """Число веток."""
        return len(self._branches)

    def children(self, branch_id: int) -> List[Dict[str, Any]]:
        """Дочерние ветки указанной ветки."""
        return [b for b in self._branches.values()
                if b.get("parent_id") == int(branch_id)]

    def list_branches(self) -> List[Dict[str, Any]]:
        """Список веток с признаком активности и числом фактов.

        Returns:
            Список словарей веток, отсортированный по id; в каждой ветке
            добавлены ключи ``active`` и ``facts_count``.
        """
        if self.persistence and self._store is not None:
            rows = self._store.load_branches()
            if rows:
                self._branches = {int(row["id"]): dict(row) for row in rows}
                if self._active not in self._branches:
                    self._active = min(self._branches)

        result: List[Dict[str, Any]] = []
        for branch_id in sorted(self._branches):
            branch = dict(self._branches[branch_id])
            branch["active"] = branch_id == self._active
            if not self.persistence:
                branch["messages"] = len(self._memory_messages.get(branch_id, []))
            branch["facts_count"] = self._facts_count(branch_id)
            result.append(branch)
        return result

    def _facts_count(self, branch_id: int) -> int:
        """Число фактов в ветке (для панели веток)."""
        facts = self.load_facts(branch_id)
        data = facts.get("data") if isinstance(facts, dict) and "data" in facts else facts
        if not isinstance(data, dict):
            return 0
        total = 0
        for values in data.values():
            if isinstance(values, dict):
                total += len(values)
        return total

    # ------------------------------------------------------------------
    # Сообщения и артефакты веток
    # ------------------------------------------------------------------

    def load_messages(self, branch_id: int) -> List[Message]:
        """Сообщения ветки (из БД, при её отсутствии — из памяти)."""
        target = int(branch_id)
        if self.persistence and self._store is not None:
            return [dict(m) for m in self._store.load_branch_messages(target)]
        return [dict(m) for m in self._memory_messages.get(target, [])]

    def remember_messages(self, branch_id: int,
                          messages: Sequence[Message]) -> None:
        """Запоминает сообщения ветки (используется без БД)."""
        self._memory_messages[int(branch_id)] = [dict(m) for m in (messages or [])]

    def load_facts(self, branch_id: int) -> Dict[str, Any]:
        """Факты (артефакты) ветки."""
        target = int(branch_id)
        if self.persistence and self._store is not None:
            raw = self._store.load_branch_facts(target)
            if isinstance(raw, dict):
                return raw
        return copy.deepcopy(self._memory_facts.get(target, {}))

    def remember_facts(self, branch_id: int, facts: Any) -> None:
        """Сохраняет факты ветки (в памяти и, при наличии, в БД)."""
        target = int(branch_id)
        plain = facts_to_plain(facts)
        self._memory_facts[target] = copy.deepcopy(plain)
        if self.persistence and self._store is not None:
            self._store.save_branch_facts(target, plain)

    # ------------------------------------------------------------------
    # Создание веток
    # ------------------------------------------------------------------

    def _auto_name(self) -> str:
        """Свободное автоимя вида «ветка-N»."""
        used = set()
        for branch in self._branches.values():
            match = _AUTO_NAME_RE.match(str(branch.get("name") or ""))
            if match:
                used.add(int(match.group(1)))
        index = 2
        while index in used:
            index += 1
        return f"ветка-{index}"

    def create(self, messages: Sequence[Message],
               artifacts: Optional[Dict[str, Any]] = None,
               from_index: Optional[int] = None,
               name: Optional[str] = None,
               strategy: Optional[str] = None) -> Dict[str, Any]:
        """Создаёт ветку от текущего места или от выбранного сообщения.

        Args:
            messages: история активной ветки (system + диалог).
            artifacts: артефакты активной ветки (facts, summary).
            from_index: индекс сообщения диалога (без system), от которого
                ответвляемся; None — checkpoint в конце (копируется весь диалог).
            name: имя ветки (None — автоимя «ветка-N»).
            strategy: подстратегия ветки (None — как у текущей ветки).

        Returns:
            Словарь {"branch", "messages", "artifacts", "strategy"}.

        Raises:
            AgentConfigError: если достигнут лимит веток.
        """
        if self.count() >= self.max_branches:
            raise AgentConfigError(
                f"Достигнут лимит веток ({self.max_branches}). "
                "Удалите ненужные ветки или увеличьте BRANCH_MAX."
            )

        system, dialog = _split_system(messages)
        if from_index is None:
            fork_at = len(dialog)
        else:
            try:
                fork_at = int(from_index)
            except (TypeError, ValueError):
                fork_at = len(dialog)
            fork_at = max(0, min(fork_at, len(dialog)))
        prefix: List[Message] = system + dialog[:fork_at]

        current = self.get_active()
        branch_strategy = str(strategy or current.get("strategy")
                              or self.default_strategy)
        branch_name = str(name or "").strip() or self._auto_name()
        new_artifacts = copy_artifacts(artifacts, len(prefix) - len(system))

        new_id: Optional[int] = None
        if self.persistence and self._store is not None:
            new_id = self._store.save_branch(
                name=branch_name,
                parent_id=self._active,
                forked_at=fork_at,
                strategy=branch_strategy,
                messages=prefix,
            )
        if new_id is None:
            new_id = max(self._branches) + 1 if self._branches else ROOT_BRANCH_ID

        branch = self._virtual_branch(new_id, branch_name, self._active)
        branch["forked_at"] = fork_at
        branch["strategy"] = branch_strategy
        branch["messages"] = len(prefix)
        self._branches[new_id] = branch
        self._memory_messages[new_id] = prefix
        self.remember_facts(new_id, new_artifacts.get("facts"))

        self._active = new_id
        if self.persistence and self._store is not None:
            self._store.set_active_branch(new_id)

        logger.info(
            "Создана ветка %r (id=%d) от ветки %r, сообщений: %d.",
            branch_name, new_id, current.get("name"), len(prefix),
        )
        return {
            "branch": branch,
            "messages": [dict(m) for m in prefix],
            "artifacts": new_artifacts,
            "strategy": branch_strategy,
        }

    def checkpoint(self, messages: Sequence[Message],
                   artifacts: Optional[Dict[str, Any]] = None,
                   name: Optional[str] = None) -> Dict[str, Any]:
        """Создаёт checkpoint — ветку от конца текущего диалога."""
        return self.create(messages, artifacts, from_index=None, name=name)

    # ------------------------------------------------------------------
    # Переключение и правка
    # ------------------------------------------------------------------

    def switch(self, branch_id: int) -> Dict[str, Any]:
        """Делает ветку активной (сообщения загружает вызывающий код).

        Args:
            branch_id: идентификатор целевой ветки.

        Returns:
            Словарь {"branch", "strategy"}.

        Raises:
            AgentConfigError: если ветки с таким id нет.
        """
        target = self.get(branch_id)
        if target is None:
            raise AgentConfigError(f"Ветка id={branch_id} не найдена.")
        self._active = int(branch_id)
        if self.persistence and self._store is not None:
            self._store.set_active_branch(self._active)
        logger.info("Активная ветка: %r (id=%d).",
                    target.get("name"), self._active)
        return {"branch": target, "strategy": target.get("strategy")}

    def rename(self, branch_id: int, name: str) -> Dict[str, Any]:
        """Переименовывает ветку.

        Raises:
            AgentConfigError: если ветки нет или имя пустое.
        """
        branch = self.get(branch_id)
        if branch is None:
            raise AgentConfigError(f"Ветка id={branch_id} не найдена.")
        new_name = str(name or "").strip()
        if not new_name:
            raise AgentConfigError("Имя ветки не может быть пустым.")
        branch["name"] = new_name
        branch["updated_at"] = _now()
        if self.persistence and self._store is not None:
            self._store.save_branch(
                name=new_name,
                parent_id=branch.get("parent_id"),
                forked_at=int(branch.get("forked_at") or 0),
                strategy=str(branch.get("strategy") or self.default_strategy),
                branch_id=int(branch["id"]),
            )
        return branch

    def delete(self, branch_id: int, cascade: bool = False) -> Dict[str, Any]:
        """Удаляет ветку (при ``cascade`` — вместе с дочерними).

        Args:
            branch_id: идентификатор удаляемой ветки.
            cascade: True — удалить и вложенные ветки.

        Returns:
            Словарь {"deleted", "active"}.

        Raises:
            AgentConfigError: ветки нет, это последняя ветка, или у ветки
                есть дочерние, а ``cascade`` не запрошен.
        """
        branch = self.get(branch_id)
        if branch is None:
            raise AgentConfigError(f"Ветка id={branch_id} не найдена.")
        if self.count() <= 1:
            raise AgentConfigError("Нельзя удалить последнюю ветку.")

        subtree = [int(branch_id)]
        if cascade:
            stack = [int(branch_id)]
            while stack:
                current = stack.pop()
                for child in self.children(current):
                    child_id = int(child["id"])
                    if child_id not in subtree:
                        subtree.append(child_id)
                        stack.append(child_id)
        else:
            kids = self.children(branch_id)
            if kids:
                names = ", ".join(str(k.get("name")) for k in kids)
                raise AgentConfigError(
                    f"У ветки «{branch.get('name')}» есть дочерние ветки "
                    f"({names}). Удалите их отдельно или разрешите каскадное "
                    "удаление."
                )

        if self.persistence and self._store is not None:
            self._store.delete_branch(int(branch_id), cascade=cascade)

        for target in subtree:
            self._branches.pop(target, None)
            self._memory_messages.pop(target, None)
            self._memory_facts.pop(target, None)

        if self._active in subtree:
            parent = branch.get("parent_id")
            if parent in self._branches:
                self._active = int(parent)
            else:
                self._active = min(self._branches)
            if self.persistence and self._store is not None:
                self._store.set_active_branch(self._active)

        logger.info(
            "Удалены ветки: %s. Активная ветка: %r.",
            ", ".join(str(t) for t in subtree), self.get_active().get("name"),
        )
        return {"deleted": subtree, "active": self.get_active()}

    def set_active(self, branch_id: int) -> Dict[str, Any]:
        """Синоним :meth:`switch` (для читаемости вызовов)."""
        return self.switch(branch_id)

    def describe(self) -> Dict[str, Any]:
        """Краткое описание состояния веток (для статистики/UI)."""
        active = self.get_active()
        return {
            "id": int(active.get("id", self._active)),
            "name": active.get("name"),
            "strategy": active.get("strategy"),
            "forked_at": active.get("forked_at"),
            "messages": int(active.get("messages") or 0),
            "branches_count": self.count(),
        }
