# Платформа обучения Android-разработке (ai_1_36)

Вкладка **«🎓 Обучение»** Flet-приложения: мок-авторизация, курс
«5 тем × 3–4 подтемы» с ленивой генерацией, проверка практических
заданий через filesystem MCP, PDF-конспекты и напоминания через
MCP reminder.

## Экраны

| Экран | Что делает |
|---|---|
| Вход | Логин/пароль, 3 мок-пользователя (SHA-256 в БД) |
| Дашборд | Приветствие, уровень, 🔥 streak, сводка прогресса, 5 тем со статусами |
| Тема | Описание, подтемы со статусами, кнопка «📥 PDF-конспект» |
| Подтема | Теория (Markdown), задание + рубрика, проверка по папке проекта, фидбек AI, отметка «выполнено» |

## Мок-пользователи

| Логин | Пароль | Уровень |
|---|---|---|
| ivan | ivan123 | beginner |
| maria | maria123 | intermediate |
| alex | alex123 | advanced |

Уровень влияет на промпт генерации темы (`LEVEL_GUIDANCE` в
`app/learning/plan.py`): beginner — проще и пошагово, advanced —
продвинутые приёмы.

## Архитектура

```
app/ui/learning_view.py   — вкладка (вход → дашборд → тема → подтема)
app/learning/db.py        — SQLite data/learning.db (users/topics/
                            subtopics/progress), мок-пользователи,
                            streak (compute_streak)
app/learning/llm.py       — DeepSeekJsonLLM (response_format=json_object,
                            ретрай с удвоением max_tokens) + MockJsonLLM
                            (--mock/офлайн) + build_json_llm(config)
app/learning/plan.py      — PlanService: ленивая генерация темы + reminder
app/learning/review.py    — ReviewService: fs MCP (tree → выбор файлов
                            LLM → read_multiple_files → оценка по рубрике)
app/learning/pdf.py       — PDFService: HTML+CSS → PDF (WeasyPrint)
```

Сборка — в `ChatApp.__init__` (app/main.py): `get_learning_settings`
путь БД (`LEARNING_DB_PATH`, по умолчанию `data/learning.db`) и папку
PDF (`LEARNING_PDF_DIR`, по умолчанию `output/pdf`). Ошибка сборки не
ломает приложение — вкладка показывает «Модуль обучения недоступен».

## Ключевые решения

### Ленивая генерация плана

При входе пользователя `ensure_topics` создаёт 5 «пустых» строк тем
(`generated_at IS NULL`). При **первом открытии** темы
`PlanService.get_or_generate_topic` одним JSON-запросом генерирует
описание + 3–4 подтемы (теория/задание/рубрика) с учётом уровня и
сохраняет в БД. Повторный вход читает из БД — LLM не вызывается.
Весь план сразу не генерируется никогда (ТЗ).

### Проверка задания — свой fs-сервер на время проверки

Общий fs-сервер приложения (порт 8005) имеет `ALLOWED_ROOT` = корень
этого проекта, а папка Android-проекта пользователя лежит **снаружи**.
`EphemeralFsClient` (app/learning/review.py) поднимает **эфемерный
экземпляр того же fs_server.py** с `--root <папка проекта>` на
свободном порту, проверка идёт по HTTP MCP:

```
tree(path=".", depth=3) → LLM выбирает 3–5 файлов (JSON {"files"})
→ read_multiple_files → LLM оценивает по заданию И рубрике
→ {"score": 1..5, "feedback"} → progress (оценка ≥ 4 завершает подтему)
```

Сервер останавливается после проверки; доступ ограничен ровно папкой
пользователя. Без рубрики проверка не запускается (ТЗ «ЧЕГО НЕ
ДЕЛАТЬ»).

### Напоминания

После генерации темы — `add_reminder(text, interval_minutes=4320,
repeat=False)` в существующий reminder-сервер («вернись к теме через
3 дня»). Best-effort: недоступный сервер не ломает генерацию —
результат содержит `reminder_status` с причиной.

### PDF через WeasyPrint

`PDFService` собирает HTML+CSS (тема, теория, задания, рубрики,
фидбек AI по выполненным; содержимое LLM экранируется) и рендерит
локально. WeasyPrint — **опциональная** зависимость; на Windows ему
нужен GTK runtime **с Pango ≥ 1.44**:

```
pip install weasyprint
```

GTK runtime (актуальный способ): «GTK3-Runtime Win64» со страницы
<https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases>
(файл `gtk3-runtime-*-ts-win64.exe`, Pango 1.50).

Частые ловушки:

- **`winget install GTK.GTK3` больше не работает** — пакета нет в
  winget (старая инструкция ставила именно его);
- **`winget install GtkD.GtkPlusRuntime.x64` не подходит** — в этом
  рантайме Pango 1.43, WeasyPrint падает
  (`function/symbol 'pango_context_set_round_glyph_positions' not
  found … error 0x7f`);
- после установки рантайма приложение нужно **перезапустить**.

Рантайм в стандартном каталоге `C:\Program Files\GTK3-Runtime Win64`
подхватывается автоматически (WeasyPrint регистрирует его сам,
приложение дополнительно поднимает его в PATH процесса —
`ensure_gtk_on_path`, работает и для per-user установки без прав
администратора). Нестандартный каталог — переменная окружения
`GTK_RUNTIME_BIN`. Без WeasyPrint/GTK приложение работает: кнопка PDF
показывает подсказку по установке. Файл:
`output/pdf/summary_<topic_id>_<YYYYMMDD>.pdf`.

## Прогресс и streak

`progress` (UNIQUE(user_id, subtopic_id)): статус, фидбек, оценка,
`completed_at`. Оценка ≥ 4 автоматически завершает подтему; есть и
ручная отметка «✅ Отметить выполненной». Streak — дни подряд с
завершениями (серия жива, если последний день — сегодня или вчера),
чистая функция `compute_streak`.

## Конфигурация (.env / config.py)

| Ключ | По умолчанию | Смысл |
|---|---|---|
| `LEARNING_DB_PATH` | `data/learning.db` | БД платформы (относительный — от корня) |
| `LEARNING_PDF_DIR` | `output/pdf` | Папка PDF-конспектов |

JSON-LLM использует общие `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` /
`DEFAULT_MODEL`; `MOCK_MODE=1` (или `--mock`) включает `MockJsonLLM` —
детерминированные план/проверка без сети (мок выбирает файлы, парся
дерево из промпта, поэтому мок-проверка работает на любом проекте).

## Тесты

```
python -m unittest tests.test_learning_platform -v   # БД, сервисы, PDF, JSON, живой fs-цикл
python -m unittest tests.test_learning_ui -v         # вкладка офлайн
```

Живой интеграционный тест (`EphemeralFsReviewIntegrationTest`)
поднимает реальный fs_server.py над tmp-проектом и проходит полный
цикл проверки с мок-LLM (сеть наружу не трогается; требует SDK `mcp`).
