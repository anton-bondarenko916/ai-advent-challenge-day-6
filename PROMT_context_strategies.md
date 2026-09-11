# ПРОМПТ ДЛЯ CODE ASSISTANT: 3 СТРАТЕГИИ УПРАВЛЕНИЯ КОНТЕКСТОМ + ВЕТКИ

## ЦЕЛЬ ЗАДАЧИ
Доработать приложение **DeepSeek Chat** (Python 3.9+ / Tkinter / DeepSeek API,
SQLite-персистентность, подсчёт токенов): добавить в агента **три стратегии
управления контекстом** и **переключатель** между ними:

1. **Sliding Window** — в запрос уходят только последние N сообщений;
2. **Sticky Facts / Key-Value Memory** — отдельный блок `facts` (ключ→значение),
   который обновляется после каждого хода и отправляется в запрос вместе с
   последними N сообщениями;
3. **Branching (ветки диалога)** — checkpoint, две ветки от одного места,
   независимое продолжение, переключение между ветками.

Изменять существующие файлы **на месте**, не переписывать проект; сохранить
работоспособность персистентности, подсчёта токенов, поля ввода, mock-режима,
логики «ошибка API → откат сообщения → Повторить».

## ИСХОДНОЕ СОСТОЯНИЕ ПРОЕКТА (учесть, не ломать)
```
deepseek-chat-app/
├── main.py            # load_config → create_agent(config) → ChatApplication → mainloop
├── src/
│   ├── config.py      # DEFAULTS + _coerce, get_prices(), get_summary_settings(); PROJECT_ROOT
│   ├── agent.py       # DeepSeekAgent: RLock, _message_history, _build_context_messages(),
│   │                  # send_message(), _update_token_stats(), get_token_stats(), _persist(),
│   │                  # _load_context(), clear_history(), update_config(), close(),
│   │                  # сводка: _summary/_summary_settings/_maybe_summarize()/_call_service_api()
│   ├── summarizer.py  # ContextSummarizer: SUMMARY_MARKER, should_summarize, build_request,
│   │                  # summarize, truncate  (код остаётся, но в переключатель не входит)
│   ├── storage.py     # ContextStore (sqlite3): таблица messages + meta(JSON:
│   │                  # params, stats, summary, schema_version="1")
│   ├── tokens.py      # TokenCounter (tiktoken/heuristic): count_text/count_messages
│   ├── ui.py          # ChatApplication: чат (теги сообщений), панель параметров
│   │                  # (модель/темп/max tokens/промпт/блок «Токены»/кнопка «Контекст…»),
│   │                  # окно «Контекст», поле ввода, статус-бар, worker+queue
│   └── utils.py       # валидаторы, format_tokens, get_logger, humanize_api_error
├── scripts/dialog_benchmark.py  # сценарии short/long/over_limit/long_with_summary (sim/live)
└── tests/             # 112 тестов (agent, utils, persistence, tokens, summarization,
                       # ui_input, benchmark, startup, gui_helpers — общий Tk-root)
```
Ключевые факты, на которые опираться:
- `_build_context_messages()` собирает то, что уходит в API; сейчас поддерживает
  режим summary (`_summary["covered"]` + system-сообщение с конспектом);
- `send_message()`: сжатие (если включено) → добавил user → снимок контекста →
  `_call_api()` → добавил assistant → `_trim_history()` (только без сжатия) →
  `_update_token_stats()` → `_persist()`;
- `_call_service_api(request)` — служебный вызов API (используется для сжатия);
  в тестах и sim-режиме подменяется;
- `get_token_stats()` отдаёт `{request, response, context{messages_raw,messages_sent,…},
  summary{…}, session{…, summary_*}, counter}` — этим живут UI и бенчмарк;
- `ContextStore`: `messages(role, content, created_at)` + `meta(key,value)`,
  `save_state(messages, params, stats=None, summary=None)`, `load_state()`,
  `clear()`, `SCHEMA_VERSION="1"`;
- `create_agent(config)` в main.py — точка сборки (покрыта `tests/test_startup.py`);
- среда: **нет PyPI и нет сети из кода**; тесты офлайн (mock/`unittest.mock`);
  GUI-тесты используют общий Tk-root (`tests/gui_helpers.py`) — второе окно
  в процессе не получает клавиатурные события; Windows-песочница запрещает
  системный TEMP и удаление каталогов (тестовые файлы — `tests/tmp*/`).

## РЕШЕНИЯ, СОГЛАСОВАННЫЕ С ЗАКАЗЧИКОМ (реализовать именно так)
1. **В переключателе ровно три стратегии:** `sliding_window`, `sticky_facts`,
   `branching`. Суммаризация (summary) **выключается** в приложении: код
   `src/summarizer.py` и его тесты остаются, агент по-прежнему принимает
   `summary_settings` для обратной совместимости, но `main.py` их больше
   не передаёт, и в UI стратегии summary нет.
2. **Sliding Window отбрасывает только в запросе**: в API уходят последние N
   сообщений, но история остаётся в памяти, БД и в чате (её видно и можно
   вернуть, переключив стратегию).
3. **Facts обновляются LLM-запросом после каждого хода** (после ответа
   ассистента, по паре «сообщение пользователя + ответ»), т.е. +1 служебный
   вызов на ход; перед следующим запросом facts уже актуальны.
4. **Формат facts — плоский key→value с категориями** (`goal`, `constraints`,
   `preferences`, `decisions`, `agreements`): новое значение перезаписывает
   старое, `null`/пустая строка удаляет ключ, неизвестные категории
   игнорируются, есть лимиты по числу фактов и по токенам.
5. **Ветки создаются от текущего места и от любого выбранного сообщения**
   в чате (клик правой кнопкой по сообщению → «Создать ветку отсюда»).
6. **Ветки хранятся в SQLite**: таблица `branches` (дерево через `parent_id`)
   + `messages.branch_id`; миграция старой БД без потери истории; активная
   ветка восстанавливается после перезапуска.
7. **UI**: выпадающий список стратегий на панели параметров + панель/окно
   «Ветки» (создать/переключить/переименовать/удалить) + индикатор текущей
   ветки и стратегии в статус-баре; блок «Токены» показывает факты.
8. **У каждой ветки своя стратегия и свои артефакты** (facts; summary — если
   когда-то включён через API): при переключении ветки восстанавливаются
   и стратегия, и артефакты.

> Трактовка, зафиксированная в промпте (важно): «branching» — это **режим
> работы с ветками**, а не отдельный способ сборки контекста. Внутри каждой
> ветки действует своя **подстратегия** из двух контекстных
> (`sliding_window` | `sticky_facts`), хранимая в строке ветки. В выпадающем
> списке три значения: выбор `branching` включает панель веток и оставляет
> подстратегию ветки; выбор `sliding_window`/`sticky_facts` меняет
> подстратегию активной ветки (и запоминается для неё).

## ТРЕБОВАНИЯ К РЕАЛИЗАЦИИ

### 1. Новый модуль `src/strategies.py` — стратегии контекста
- Базовый интерфейс (имена на усмотрение, но контракт такой):
  ```python
  class ContextStrategy:
      name: str
      def build_context(self, system_prompt: str, dialog: list, artifacts: dict) -> list
      def after_turn(self, user_text: str, assistant_text: str, dialog: list,
                     artifacts: dict, call_service) -> dict   # обновление артефактов
      def describe(self) -> dict                                # для UI/статистики
  ```
- **`SlidingWindowStrategy`**: `build_context` = `[system] + dialog[-N:]`
  (`WINDOW_KEEP_LAST_N`, по умолчанию 20); `after_turn` — ничего не делает,
  артефактов нет.
- **`StickyFactsStrategy`**: `build_context` = `[system] + [system: блок facts] +
  dialog[-N:]`; `after_turn` — служебный вызов API, который получает текущие
  facts + новый обмен и возвращает обновлённый JSON facts.
- **Реестр/factory**: `STRATEGIES = {"sliding_window": …, "sticky_facts": …}`,
  `create_strategy(name, settings, deps)`; неизвестное имя → `AgentConfigError`
  со списком доступных. Модуль не импортирует tkinter.
- Стратегия **не хранит** состояние сама: facts живут в `FactsStore` (п. 2),
  история — в агенте; стратегия только собирает контекст и обновляет артефакты.
- Ошибка служебного обновления facts **не должна** ломать ход: вернуть прежние
  facts + заполнить `error` (по аналогии с `_run_summarization`).

### 2. Новый модуль `src/facts.py` — блок key→value
- Класс `FactsStore` (или набор чистых функций + класс-обёртка):
  - хранение: `{category: {key: value}}`, категории из фиксированного набора;
  - `merge(current, incoming)` — новое значение перезаписывает старое,
    `null`/пустая строка удаляет ключ, чужие категории/не-строковые значения
    отбрасываются с предупреждением в лог;
  - `to_messages()` — текст для system-сообщения вида
    `Известные факты о диалоге:\n[goal] цель: …\n[preferences] любимый_цвет: синий`;
  - `token_count(counter)`, `limit(counter, max_items, max_tokens)` — при
    превышении лимитов удаляются самые старые/наименее важные факты
    (по `updated_at`, затем по категории), с записью в лог;
  - `remember(key, value, category)`, `forget(key)`, `clear()`, `as_dict()`,
    `from_dict(raw)` — устойчиво к мусору в JSON (не падать, предупреждать);
  - `extract_local(text)` — локальный fallback: разбор явных команд
    («запомни, что …», «запомни: …», «меня зовут …», «мой … — …»);
    используется, если LLM-обновление недоступно/упало, и как «гарантия»
    для явных команд пользователя.
- **Служебный промпт извлечения** (шаблон, переопределяется `FACTS_PROMPT`)
  обязан содержать маркер `FACTS_MARKER = "[[ОБНОВЛЕНИЕ_ФАКТОВ]]"` (по аналогии
  с `SUMMARY_MARKER`: по нему служебный запрос отличается в тестах, логах и
  sim-режиме бенчмарка) и требовать **строго JSON** вида
  `{"goal": {...}, "constraints": {...}, "preferences": {...}, "decisions": {...}, "agreements": {...}}`
  с полным обновлённым набором фактов (модель обязана сохранить прежние факты,
  если они не устарели).
- Парсинг ответа: вырезать JSON из текста (в т.ч. из ```-блока), при ошибке —
  оставить прежние facts, записать warning/`error`, не ронять ход.

### 3. Новый модуль `src/branches.py` — дерево веток
- Класс `BranchManager` поверх `ContextStore` (или расширение хранилища):
  - `list_branches()`, `get_active()`, `get(branch_id)`;
  - `create(from_index: Optional[int] = None, name: Optional[str] = None,
    strategy: Optional[str] = None)` — создаёт ветку: копирует префикс
    сообщений текущей ветки (`from_index=None` — весь диалог, т.е. checkpoint
    в конце), копирует артефакты (facts; summary, если он есть и `covered`
    не больше длины префикса), ставит `parent_id`, `forked_at`, `strategy`;
    автоимя вида `ветка-2`, если имя не задано;
  - `switch(branch_id)` — сохраняет состояние текущей ветки, загружает целевую
    (сообщения + артефакты + стратегию);
  - `rename(branch_id, name)`, `delete(branch_id, cascade=False)` — запрещено
    удалять последнюю ветку; ветка с детьми без `cascade=True` → понятная
    ошибка; удаление активной ветки переключает на родителя;
  - `checkpoint(name=None)` — удобный вызов `create(from_index=None)`;
  - лимит `BRANCH_MAX` (по умолчанию 20) с понятной ошибкой.
- **Физическое копирование префикса** (без сложной логики «общего предка»):
  история в проекте ограничена, это проще и надёжнее; в docstring объяснить выбор.
- Ветки не зависят от tkinter; ошибки БД не пробрасываются наружу (лог + `error`).

### 4. `src/storage.py` — схема v2 и миграция
- Добавить таблицу:
  ```sql
  CREATE TABLE branches (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      parent_id INTEGER NULL,
      forked_at INTEGER NOT NULL DEFAULT 0,
      strategy TEXT NOT NULL DEFAULT 'sliding_window',
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
  );
  ```
- В `messages` добавить `branch_id INTEGER NOT NULL DEFAULT 1` и `seq INTEGER`
  (порядок внутри ветки). `SCHEMA_VERSION = "2"`.
- **Миграция v1 → v2 (в одной транзакции, без потери данных):** создать
  `branches`; вставить корневую ветку `id=1, name='main', strategy=<STRATEGY из конфига>`;
  `ALTER TABLE messages ADD COLUMN …`; проставить `branch_id=1` и `seq` по
  порядку `id`; записать `schema_version="2"`. При сбое — откат, лог-ошибка и
  работа в режиме одной истории (как раньше), без падения приложения.
- API хранилища расширить обратносовместимо:
  - `load_branches()`, `save_branch(...)`, `delete_branch(...)`,
    `set_active_branch(id)`, `get_active_branch()`;
  - `save_state(messages, params, stats=None, summary=None, branch_id=None,
    facts=None)` — старые вызовы без новых аргументов продолжают работать
    (пишут в активную ветку), `facts=None` не затирает сохранённые;
  - `load_state()` возвращает дополнительно `"facts"`, `"branches"`,
    `"active_branch"` (отсутствие ключей = «нет данных», не ошибка).
- Все новые ключи — в `meta` (`active_branch`, `facts`) и в таблице сообщений;
  старые БД читаются, существующие тесты персистентности остаются зелёными.

### 5. `src/agent.py` — переключатель стратегий и ветки
- **Новые параметры `__init__`**: `strategy: Optional[str] = None`
  (имя стратегии; None → из настроек/по умолчанию `sliding_window`),
  `strategy_settings: Optional[Dict[str, Any]] = None` (N окна, факты, ветки),
  `branch_manager=None` (для тестов).
- **Состояние**: `self._strategy_name`, `self._strategy` (объект),
  `self._facts: Dict[str, Dict[str, str]]`, `self._active_branch_id`,
  `self._facts_error`, `self._facts_updated_at`, служебные счётчики facts в
  `_session_stats`: `facts_requests`, `facts_prompt_tokens`,
  `facts_completion_tokens`, `facts_estimated_tokens`, `facts_cost_usd`.
- **Сборка контекста**: `_build_context_messages()` делегирует стратегии
  (`strategy.build_context(...)`); режим summary сохраняется только для
  обратной совместимости, когда `summary_settings` передан явно и стратегия
  не выбрана (в приложении не используется).
- **`send_message()`**: после успешного ответа вызывать
  `strategy.after_turn(user_text, assistant_text, dialog, artifacts, self._call_service_api)`
  (для `sticky_facts` это LLM-обновление facts), обновлять счётчики facts,
  сохранять артефакты (`_persist()`), логировать; при ошибке facts — оставить
  прежние, записать `_facts_error`, ход не ломать, повтор при следующем ходу.
- **Публичное API**:
  - `available_strategies() -> list[str]` (`sliding_window`, `sticky_facts`, `branching`);
  - `get_strategy() -> dict` (`{name, window, facts_count, facts_tokens, branch, available, error}`);
  - `set_strategy(name) -> dict` — валидация, применение, сохранение в ветке/`meta`;
    при переходе на `sticky_facts` выполнить **начальное извлечение фактов**
    по текущему диалогу (чтобы facts не были пустыми до следующего хода);
  - `get_facts() -> dict`, `remember(key, value, category="decisions")`,
    `forget(key)`, `clear_facts()`, `update_facts_now(force=True) -> dict`;
  - ветки: `list_branches()`, `get_active_branch()`, `create_branch(from_index=None,
    name=None, strategy=None)`, `switch_branch(branch_id)`, `rename_branch(...)`,
    `delete_branch(branch_id, cascade=False)`, `checkpoint(name=None)`;
  - `get_token_stats()` расширить: блоки `strategy` (имя, окно, режим),
    `facts` (count, tokens, updated_at, error), `branch` (id, name, strategy,
    forked_at, messages, branches_count), в `context` — как и раньше
    `messages_raw`/`messages_sent`; в `session` — счётчики `facts_*`.
- **`clear_history()`** очищает сообщения **активной ветки** (артефакты ветки —
  facts — тоже, если пользователь подтвердил «Очистить историю»); удаление
  ветки — отдельная операция. Смена стратегии `clear_history` не вызывает.
- **Персист/восстановление**: при `_load_context()` восстановить активную ветку,
  её сообщения, стратегию и facts; при `close()` — финальный flush.
- Потокобезопасность: всё под существующим RLock; служебные вызовы — через
  `_call_service_api` (подменяемы в тестах).

### 6. `src/config.py`, `.env`, `.env.example` — настройки
Добавить в `DEFAULTS` (с корректным приведением типов):
- `STRATEGY` (str, `sliding_window`) — значение по умолчанию: `sliding_window`
  | `sticky_facts` | `branching`;
- `WINDOW_KEEP_LAST_N` (int, 20) — размер окна для sliding window;
- `FACTS_ENABLED` (bool, true) — обновлять facts LLM-запросом;
- `FACTS_MODEL` (str, "") — пусто = модель диалога;
- `FACTS_TEMPERATURE` (float, 0.0);
- `FACTS_MAX_TOKENS` (int, 300) — лимит блока facts в токенах;
- `FACTS_MAX_ITEMS` (int, 50) — максимум пар ключ→значение;
- `FACTS_PROMPT` (str, "") — свой промпт извлечения;
- `BRANCH_MAX` (int, 20) — максимум веток;
- `SUMMARY_*` оставить (обратная совместимость), но `main.py` их не передаёт —
  в README указать, что суммаризация не входит в переключатель.
Добавить `get_strategy_settings(config)` с валидацией и клампингом
(некорректные значения → значения по умолчанию + WARNING), по аналогии с
`get_summary_settings()`. Описать все переменные в `.env.example` и `.env`.

### 7. `src/ui.py` — переключатель и панель веток
- **Стратегия**: `ttk.Combobox` на панели параметров (`sliding_window`,
  `sticky_facts`, `branching`) + подпись текущего окна/лимита; смена применяется
  к агенту сразу (`agent.set_strategy`), статус в статус-баре.
- **Панель «Ветки…»** (кнопка рядом с «Контекст…»): окно `Toplevel` с таблицей
  веток (`ttk.Treeview`: имя, стратегия, сообщений, фактов, создана, родитель),
  кнопками «Создать от конца» (checkpoint), «Создать от выбранного сообщения»,
  «Переключиться», «Переименовать», «Удалить» (с подтверждением) и «Закрыть»;
  активная ветка выделена.
- **Ветка от сообщения**: у каждого сообщения в чате — свой тег с индексом;
  правый клик по сообщению → контекстное меню «Создать ветку отсюда…»
  (диалог ввода имени) → `agent.create_branch(from_index=i)` + переключение.
- **Окно «Контекст»** переделать под новую модель: показывает стратегию, окно,
  блок facts (категория/ключ/значение, прокрутка) и информацию о ветке; кнопки
  «Обновить факты» (фоновый поток, статус «Обновляю факты…»), «Очистить факты»,
  «Закрыть»; для `sliding_window` — понятное «факты не используются в этой
  стратегии».
- **Блок «Токены»**: вместо «Конспект: …» показывать
  `Факты: N · ≈X токенов` (для окна — `Окно: N сообщений`), сохранить
  `Запрос/Ответ/Контекст/Сессия` и счётчик.
- **Статус-бар**: `Ветка: main · Стратегия: sticky_facts · ctx: ≈X (Y сообщ. отправлено)`.
- Всё, что ходит в API по кнопкам, — в фоновом потоке через `queue`/`root.after`
  (как `_on_context_refresh`), UI не должен «зависать»; окна не создавать второй `Tk()`.

### 8. `main.py` — сборка
- `create_agent(config)` передаёт `strategy=get_strategy_settings(config)`;
  `summary_settings` больше не передаётся (суммаризация выключена в приложении).
- Обновить `tests/test_startup.py`: проверять выбор стратегии, окно и лимиты
  фактов вместо настроек summary (остальные стартовые проверки сохранить).

### 9. `scripts/dialog_benchmark.py` — сравнение стратегий
- Новые сценарии (офлайн, sim):
  - `window` — 30 ходов, `sliding_window` (N=20): контекст ограничен, факт из
    начала диалога теряется;
  - `facts` — 30 ходов, `sticky_facts`: факт сохраняется за счёт блока facts,
    контекст ограничен, видны служебные запросы (≈по одному на ход) и их токены;
  - `branching` — сценарий ветвления: несколько ходов → checkpoint → две ветки
    от одного места с разными продолжениями → переключение между ветками →
    проверка изоляции (в ветке A нет сообщений ветки B, и наоборот) и того,
    что каждая ветка помнит своё;
  - опция `--strategies window,facts,branching` (по умолчанию все) и сохранение
    прежних сценариев (`--no-summary` или `--with-summary` — на усмотрение,
    но legacy-сценарий summary должен оставаться доступен явным флагом).
- `SimulatedModel` доработать: распознавать служебный запрос извлечения фактов
  по `FACTS_MARKER` и возвращать валидный JSON facts (включая факт из начала
  диалога), а для ветвления — отвечать так, чтобы ответ зависел от истории
  своей ветки (например, «в этой ветке я помню: …»).
- Итоговая таблица: `Стратегия | Ходов | Контекст(макс) | Сессия | Стоимость |
  Помнит | Веток | Служебных запросов`; JSON-отчёт дополнить блоками
  `strategy`/`facts`/`branch`.

### 10. Тесты (офлайн, без сети; временные файлы — `tests/tmp*/`)
Новые файлы:
- `tests/test_strategies.py`: реестр/factory (неизвестное имя → ошибка),
  `SlidingWindowStrategy.build_context` (system первым, ровно N последних),
  `StickyFactsStrategy` (facts-сообщение в контексте, порядок system→facts→окно),
  переключение стратегии в агенте (`set_strategy`, сохранение между запусками),
  начальное извлечение фактов при переходе на `sticky_facts`,
  `context.messages_sent` для каждой стратегии, статистика `facts_*` отдельная
  от статистики диалога;
- `tests/test_facts.py`: merge (перезапись, удаление через `null`, чужие
  категории/типы отбрасываются), лимиты по items/токенам, парсинг битого JSON
  (старые facts сохраняются), локальный fallback «запомни…», публичное API
  (`remember/forget/clear_facts`), обновление после каждого хода через
  подменённый `_call_service_api`, персистентность facts в meta;
- `tests/test_branches.py`: миграция v1→v2 (старая БД → ветка `main`, история
  целиком), создание ветки от конца и от индекса, изоляция сообщений,
  переключение восстанавливает стратегию и facts, переименование, удаление
  (лист, с детьми без cascade → ошибка, последняя ветка → ошибка), лимит
  `BRANCH_MAX`, восстановление активной ветки после «перезапуска»;
- GUI-тесты (пропускать без дисплея, общий root через `tests/gui_helpers.py`):
  смена стратегии в выпадающем списке меняет стратегию агента; окно «Ветки»
  создаёт/переключает ветку; правый клик по сообщению предлагает «Создать ветку
  отсюда».
- Обязательно: `tests/test_summarization.py` и все остальные существующие тесты
  остаются зелёными (кроме оговорённого обновления `tests/test_startup.py`).

### 11. README
- Новый раздел «Стратегии управления контекстом и ветки»: таблица стратегий,
  что уходит в запрос, как обновляются facts, как работать с ветками,
  переменные окружения, ограничения (окно не удаляет историю, facts обновляются
  раз за ход, стоимость служебных запросов учитывается отдельно), примеры
  запуска бенчмарка.

## КРИТЕРИИ ПРИЁМКИ
- [ ] В агенте есть переключатель из трёх стратегий (`sliding_window`,
      `sticky_facts`, `branching`); выбор сохраняется и восстанавливается после
      перезапуска; в UI есть выпадающий список.
- [ ] Sliding Window: в API уходят последние N сообщений, история в БД/чате
      сохраняется; переключение стратегии возвращает полный контекст.
- [ ] Sticky Facts: facts обновляются после каждого хода LLM-запросом (строгий
      JSON, устойчивость к ошибкам), отправляются отдельным system-сообщением
      вместе с последними N; явные команды «запомни…» работают даже при сбое LLM.
- [ ] Branching: checkpoint, две независимые ветки от одного места,
      переключение; ветки хранятся в SQLite, переживают перезапуск; миграция
      старой БД не теряет историю.
- [ ] У каждой ветки своя стратегия и свои facts; переключение ветки
      восстанавливает их.
- [ ] Токены/стоимость служебных запросов facts учитываются отдельными
      счётчиками; UI показывает факты, ветку и стратегию.
- [ ] Бенчмарк показывает разницу стратегий (контекст/память/стоимость) и
      изоляцию веток.
- [ ] Все существующие тесты (с оговорённой правкой `test_startup.py`) + новые
      — зелёные, офлайн, без новых зависимостей.

## ЧТО НЕ ДЕЛАТЬ
- Не удалять `src/summarizer.py` и его тесты; не ломать приём `summary_settings`
  агентом (обратная совместимость), но в приложении и в переключателе summary
  не использовать.
- Не удалять историю из БД при sliding window (отбрасывается только в запросе).
- Не терять данные при миграции схемы; не менять существующие публичные
  сигнатуры ломающим образом (`save_state`/`load_state`, `send_message`,
  `get_token_stats` — только обратносовместимое расширение).
- Не выполнять сетевые вызовы в тестах; не добавлять зависимости; не хардкодить
  цены; не трогать mock-изоляцию и «ошибка API → откат → Повторить».
- Не блокировать UI служебными запросами (фоновый поток + очередь).

## ОГРАНИЧЕНИЯ СРЕДЫ РАЗРАБОТКИ
- Нет доступа к PyPI и нет сети из кода: только stdlib + уже используемые
  пакеты (openai, python-dotenv, tiktoken); тесты подменяют `_call_api` /
  `_call_service_api`.
- Windows + песочница: не использовать системный TEMP, не полагаться на
  удаление каталогов; тестовые файлы — внутри репозитория (`tests/tmp*/`).
- Tk в одном процессе доставляет клавиатурные события только первому окну —
  GUI-тесты обязаны использовать общий root из `tests/gui_helpers.py`.
- Консоль может искажать кириллицу — не ошибка приложения.

## ВЫХОДНЫЕ ДАННЫЕ
1. Список изменённых/новых файлов (diff-стиль): `src/strategies.py`,
   `src/facts.py`, `src/branches.py` (новые), `src/agent.py`, `src/storage.py`,
   `src/config.py`, `src/ui.py`, `main.py`, `scripts/dialog_benchmark.py`,
   `tests/test_strategies.py`, `tests/test_facts.py`, `tests/test_branches.py`
   (новые), `tests/test_startup.py` (обновлён), `.env`, `.env.example`, `README.md`.
2. Полный код новых и изменённых модулей.
3. Объяснение архитектуры: как выбирается стратегия, что уходит в запрос в каждой
   стратегии, как обновляются и хранятся facts, как устроено дерево веток и
   миграция БД, что происходит при ошибках/очистке/перезапуске.
4. Отчёт: прогон всех тестов (существующие + новые) и вывод бенчмарка по трём
   стратегиям, включая демонстрацию изоляции веток.
