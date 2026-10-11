# ПРОМПТ ДЛЯ CODING ASSISTANT: TELEGRAM-БОТ ДЛЯ ЛОКАЛЬНОЙ LLM (OLLAMA)

## КОНТЕКСТ

Есть рабочее Python-приложение (Flet UI) с AI-агентом, использующим
Ollama локально (`127.0.0.1:11434`). Уже реализованы: переключение
провайдеров DeepSeek/Ollama, оптимизация локальной модели, удалённый
доступ через Nginx (DDNS/Tunnel/LAN). Проект развивается итерациями по
русским ТЗ; ветки/коммиты именуются `ai_1_N` (следующая итерация —
`ai_1_31`).

В проекте есть устоявшийся паттерн «собственные сервисы рядом с
приложением»: `reminder_server.py` (FastMCP HTTP, SQLite reminders.db,
APScheduler), `news_server.py`, `geo_server.py`, `weather_server.py` —
отдельные процессы в корне проекта, со своей конфигурацией через
`.env`/`src/config.py` и офлайн-тестами. Telegram-бот делается по этому
же образцу: самостоятельный скрипт, работающий 24/7 независимо от GUI.

**Задача:** Telegram-бот, хостящийся на домашнем ПК, который принимает
сообщения пользователей в Telegram и отвечает им через локальную LLM
(Ollama). Общаться можно из любой точки мира — бот держит только
ИСХОДЯЩИЕ соединения, никакие порты наружу не открываются.

## АРХИТЕКТУРА

```text
[Телефон / ПК пользователя]
        │  Telegram (интернет)
        ▼
[Серверы Telegram]
        ▲  long polling getUpdates (ТОЛЬКО исходящий трафик с ПК)
[telegram_bot.py на вашем ПК]
        │  POST http://127.0.0.1:11434/api/chat
        ▼
[Ollama: локальная модель]
        [data/telegram_bot.db — история диалогов, по пользователю]
```

## ПРИНЯТЫЕ РЕШЕНИЯ (обсуждены с заказчиком — НЕ пересматривать)

1. **Long polling** (getUpdates), не webhook — webhook потребовал бы
   публичный HTTPS и всю обвязку Nginx/Tunnel, что лишает идею смысла.
2. **Библиотека aiogram 3.x** (`aiogram>=3.7,<4`, в requirements.txt;
   тянет aiohttp и pydantic). API именно тройки: `Bot`, `Dispatcher`,
   `Router`, фильтры `Command(...)`/`F`, `dp.start_polling(bot)`.
3. **Мозг — напрямую в Ollama** (`POST /api/chat`), НЕ через ядро
   DeepSeekAgent (оно однопользовательское). История диалогов —
   раздельная на каждого пользователя, в SQLite по образцу
   reminders.db.
4. **UX:** ответы приходят целиком (стриминг-редактирование сообщения —
   v2), индикатор «печатает…» на время генерации, разбиение длинных
   ответов по лимиту Telegram 4096 символов, команды `/start`, `/help`,
   `/clear`, `/model` (выбор модели инлайн-кнопками).
5. **Безопасность:** токен бота в `.env` (НЕ хардкодить), белый список
   Telegram user ID (`TELEGRAM_ALLOWED_IDS`); незнакомец получает
   вежливый отказ, его ID пишется в лог.

## ШАГ 1: КОНФИГУРАЦИЯ (src/config.py + .env)

Добавить в `DEFAULTS` и поддержать в `get_telegram_settings()` по
образцу `get_news_settings()` (env-переменные читает python-dotenv):

| Ключ | Дефолт | Смысл |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | `""` | токен от @BotFather; пусто — бот не стартует |
| `TELEGRAM_ALLOWED_IDS` | `""` | CSV целых чисел Telegram user ID |
| `TELEGRAM_MODEL` | `""` | модель по умолчанию; пусто — первая из /api/tags |
| `TELEGRAM_HISTORY_LIMIT` | `20` | сколько последних сообщений уходит в контекст |
| `TELEGRAM_DB_PATH` | `data/telegram_bot.db` | путь к БД (относительный — от PROJECT_ROOT через get_project_path) |

В `.env` добавить закомментированные плейсхолдеры с инструкцией
(токен у @BotFather, свой числовой ID — у @userinfobot).

## ШАГ 2: ХРАНИЛИЩЕ ИСТОРИИ (src/tg_storage.py)

SQLite, схема:

```sql
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    role       TEXT    NOT NULL,  -- 'user' | 'assistant'
    content    TEXT    NOT NULL,
    created_at TEXT    NOT NULL   -- ISO 8601
);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, id);

CREATE TABLE IF NOT EXISTS user_state (
    user_id    INTEGER PRIMARY KEY,
    model      TEXT    NOT NULL DEFAULT '',  -- выбранная модель ('' — дефолт)
    updated_at TEXT    NOT NULL
);
```

Класс `TelegramHistoryStore` (протокол: `append(user_id, role, content)`,
`load(user_id, limit) -> list[dict]` в формате сообщений Ollama
`[{"role": ..., "content": ...}]`, `clear(user_id)`, `get_model(user_id)`,
`set_model(user_id, model)`).

**КРИТИЧЕСКИЕ ЛОВУШКИ проекта:**

* короткоживущие соединения `sqlite3.connect(...)` и ОБЯЗАТЕЛЬНЫЙ
  `conn.close()` (лучше контекстный менеджер/`try-finally`) — Windows
  держит файл, пока соединение открыто, и тесты/повторный запуск падают;
* таблица создаётся лениво при первом обращении; ошибки БД НЕ должны
  ронять бота (лог + бот продолжает, ход без истории);
* `load` возвращает последние `limit` сообщений в хронологическом
  порядке (SELECT ... ORDER BY id DESC LIMIT ? затем reverse);
* подрезка: при append удалять у пользователя всё, кроме последних 200
  сообщений (защита от бесконечного роста).

## ШАГ 3: КЛИЕНТ OLLAMA (src/tg_ollama_client.py)

Асинхронный httpx-клиент (ОДИН `httpx.AsyncClient` на процесс):

```python
class OllamaChatError(RuntimeError): ...      # ошибка сервиса (поле error, 4xx/5xx)
class OllamaUnavailableError(OllamaChatError): ...  # сеть/таймаут/сервер не запущен

class OllamaChatClient:
    def __init__(self, base_url=..., model_default="", read_timeout=300.0):
        # httpx.Timeout(connect=10, read=300, write=30, pool=30) —
        # локальная модель с холодной загрузкой может думать МИНУТАМИ
    async def list_models(self) -> list[str]:
        # GET /api/tags -> [m["name"] ...]
    async def chat(self, model: str, messages: list[dict]) -> str:
        # POST /api/chat {"model":..., "messages":..., "stream": false}
        # -> j["message"]["content"]; поле "error" -> OllamaChatError
    async def aclose(self): ...
```

* `stream: false` — для MVP ответ приходит одним JSON;
* любые `httpx.HTTPError`/таймауты оборачиваются в
  `OllamaUnavailableError` (наверх превращается в дружелюбное
  сообщение пользователю, а не в падение);
* никаких блокирующих вызовов (requests, time.sleep) в обработчиках —
  только async.

## ШАГ 4: ЛОГИКА БОТА (src/tg_bot_logic.py — чистые функции, тестируемые без Telegram)

* `parse_allowed_ids(raw: str) -> frozenset[int]` — CSV с пробелами,
  мусорные элементы игнорируются с warning'ом.
* `split_message(text: str, limit: int = 4000) -> list[str]` — разбиение
  длинного ответа на части <= limit по границам слов/переносов
  (4096 — жёсткий лимит Telegram, берём запас); пустой/короткий текст —
  один элемент.
* `is_allowed(user_id, allowed) -> bool`.
* `class ChatTurn`: метод `async answer(user_id, text) -> list[str]` —
  полный ход: загрузить историю (limit) → добавить user-сообщение →
  `ollama.chat(model, messages)` → добавить assistant-сообщение →
  `split_message(ответ)`. Ошибка Ollama НЕ записывает assistant-
  сообщение в историю (но user-сообщение — записывает: вопрос не
  теряется, повторный вопрос не дублируется в контексте).
* Выбор модели: модель пользователя из `user_state`, пусто —
  `TELEGRAM_MODEL`, пусто — первая из `list_models()`.

## ШАГ 5: ТОЧКА ВХОДА (telegram_bot.py в корне проекта, по образцу news_server.py)

aiogram 3:

```python
router = Router()

@router.message(CommandStart())
async def cmd_start(message: Message) -> ...       # приветствие + модель
@router.message(Command("help"))
@router.message(Command("clear"))                   # очистка СВОЕЙ истории
@router.message(Command("model"))                   # InlineKeyboard, 2 колонки,
                                                    # callback_data="model:<name>",
                                                    # текущая модель с ✅
@router.callback_query(F.data.startswith("model:")) # выбрать, answer() колбэк,
                                                    # отредактировать сообщение
@router.message(F.text)                             # обычный ход чата
```

Правила:

* **Гейт доступа первым делом в каждом message-хендлере** (или общим
  фильтром): незнакомцу — «⛔ Нет доступа. Ваш ID: `<id>` — сообщите
  владельцу», его ID в лог (warning, БЕЗ повторов на каждое сообщение —
  дедупликация по user_id);
* индикатор: `ChatActionSender(bot=..., chat_id=..., action="typing")`
  вокруг генерации;
* ответ отправляется частями `split_message` с паузой ~0.4 с между
  чанками (flood-лимиты Telegram ~1 сообщение/с на чат);
* **parse_mode НЕ задавать** (plain text): сырой вывод модели содержит
  `<`, `*`, ``` — HTML/Markdown-парсер Telegram упадёт;
* глобальный обработчик ошибок диспетчера: traceback в лог, пользователю
  «⚠️ Внутренняя ошибка, попробуйте ещё раз»; цикл polling никогда не
  умирает (aiogram переживает ошибку хендлера; наши исключения тоже
  ловим по месту);
* запуск: `python telegram_bot.py`; без `TELEGRAM_BOT_TOKEN` — понятная
  ошибка и exit(1); лог — `logging` в файл `telegram_bot.log` (UTF-8,
  rotation не обязателен) + консоль; `bot.delete_webhook(drop_pending_updates=True)`
  перед start_polling, чтобы webhook-остатки не мешали;
* корректное завершение: `aclose()` httpx-клиента (finally / on shutdown).

## ШАГ 6: АВТОЗАПУСК (docs/telegram_bot.md)

Документ: получение токена у @BotFather, свой ID у @userinfobot,
заполнение `.env`, запуск вручную, автозапуск как задача планировщика:

```cmd
schtasks /Create /TN "Ollama Telegram Bot" /SC ONSTART /RU SYSTEM ^
  /TR "cmd /c cd /d C:\AI\AI_2_1\deepseek-chat-app && python telegram_bot.py"
```

(рабочий каталог — корень проекта, иначе не найдутся .env и БД) +
таблица команд бота и чек-лист проверки с телефона.

## ТЕСТЫ (tests/test_telegram_bot.py — офлайн, по конвенциям проекта)

1. `parse_allowed_ids`: CSV/пробелы/мусор/пустая строка.
2. `split_message`: короткий текст; ровно на границе; длинный — все
   части <= limit, склейка частей восстанавливает текст; перенос слов
   по границе слов.
3. `TelegramHistoryStore` (временная БД): append/load (хронология и
   лимит), clear, get/set_model, ПЕРЕЗАПИСЬ объекта store — история
   жива; подрезка >200.
4. `OllamaChatClient` (httpx MockTransport): chat OK (content извлечён),
   поле error -> OllamaChatError, таймаут/сеть -> OllamaUnavailableError,
   list_models.
5. `ChatTurn.answer` (клиент замокан): история дополнена с двух сторон,
   в модель ушли последние N сообщений, ответ разбит на части; при
   ошибке Ollama — user-сообщение в истории есть, assistant — нет.
6. Гейт доступа: `is_allowed` False для постороннего.
7. Всё через `unittest`, запуск `python -m unittest discover -s tests`
   из корня — полный набор проекта должен остаться зелёным.

## КРИТЕРИИ ПРИЁМКИ

✅ `python telegram_bot.py` поднимает бота (long polling, порты наружу
не открываются — проверить `netstat -an | findstr LISTENING`)
✅ Посторонний пользователь получает отказ + его ID в лог
✅ Диалоги раздельные по пользователям, переживают перезапуск бота,
`/clear` сбрасывает
✅ `/model` показывает установленные модели (из /api/tags) инлайн-кнопками,
выбор сохраняется и используется в следующих ходах
✅ Ответ > 4096 символов приходит несколькими сообщениями
✅ Во время генерации виден индикатор «печатает…»
✅ Остановленный Ollama: пользователь получает дружелюбное сообщение,
бот продолжает работать (цикл не умирает)
✅ Токен только из .env; в коде/логах ключей нет
✅ Все офлайн-тесты проекта проходят (набор растёт на новые тесты)

## ЧЕГО НЕ ДЕЛАТЬ

❌ Webhook (нужен публичный HTTPS — теряется весь смысл long polling)
❌ Хардкодить токен/ID (только .env; токен не логировать)
❌ Блокирующие вызовы в хендлерах (requests, time.sleep, sync-клиент) —
   вешается весь polling
❌ Отправлять полную историю без лимита — контекст модели переполнится
❌ parse_mode HTML/Markdown на сыром выводе модели
❌ Отвечать незнакомцам повтором их текста (эхо-утечка) — только
   стандартный отказ
❌ Держать открытым SQLite-соединение (Windows залочит файл)
❌ Слать чанки ответа без паузы — flood-limit Telegram
❌ Хранить историю только в памяти — перезапуск бота не должен терять
   диалоги

## ВНЕ ОБЛАСТИ (идеи на потом, сейчас НЕ делать)

* потоковое редактирование сообщения по мере генерации (editMessageText
  с троттлингом ~1/с);
* голосовые сообщения (whisper), картинки;
* мост к ядру приложения (RAG, MCP-инструменты, память задачи);
* кнопка «Запустить/остановить бота» в панели приложения;
* DeepSeek как резервный провайдер, когда локальная модель не запущена.
