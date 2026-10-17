# Серверы приложения DeepSeek Chat

Документ описывает все серверы и сетевые сервисы, которые использует
приложение: локальные MCP-серверы (HTTP и stdio), внешние API и
RSS-источники.

Правило безопасности: MCP-серверы и Ollama поднимаются **только** на
локальных адресах (`127.0.0.1` / `localhost` / `::1`) — хостить их в
интернете нельзя (см. комментарии в `src/config.py` и
`docs/remote_access.md`).

---

## 1. Локальные MCP-серверы (HTTP, FastMCP, 24/7)

Все они запускаются из корня проекта как отдельные процессы, общаются по
MCP поверх HTTP, эндпоинт `http://<host>:<port>/mcp`. Один и тот же
механизм запуска — `src/reminder_launcher.py` (`HttpMCPServerLauncher`):
проверка порта → если свободен, старт в **новом окне консоли**
(`CREATE_NEW_CONSOLE`), если занят — процесс не дублируется. Серверы
живут независимо от приложения: закрытие чата их не останавливает.
Автозапуск при старте — `launch_mcp_servers()` в `app/main.py` (и в
Tkinter-версии `src/ui.py`); если сервер упал, HTTP-клиент сам поднимает
его хуком `ensure_running()`.

| Сервер | Скрипт | Порт | Вкл/выкл | Launcher | Инструменты |
|---|---|---|---|---|---|
| Напоминания | `reminder_server.py` | 8000 | `REMINDER_ENABLED` | `ReminderServerLauncher` | `add_reminder`, `delete_reminder`, `get_all_reminders`, `get_due_reminders`, `get_stats` |
| Новости | `news_server.py` | 8001 | `NEWS_ENABLED` | `NewsServerLauncher` | `search_news`, `summarize_text`, `save_to_file` |
| Геолокация | `geo_server.py` | 8002 | `GEO_ENABLED` | `GeoServerLauncher` | `get_user_city` |
| Погода | `weather_server.py` | 8003 | `WEATHER_ENABLED` | `WeatherServerLauncher` | `get_weather` |
| CRM (поддержка) | `crm_server.py` | 8004 | `CRM_ENABLED` | `CrmServerLauncher` | пользователи и тикеты для команды `/support` |
| Файловая система | `fs_server.py` | 8005 | `FS_ENABLED` | `FsServerLauncher` | чтение/запись/поиск/дерево по файлам проекта (корень — сам проект, `FS_ROOT` переопределяет) |

Настройки (`src/config.py`, `.env.example`):

* `*_ENABLED` — выключить сервер и его инструменты;
* `*_SERVER_HOST` (по умолчанию `127.0.0.1`) и `*_SERVER_PORT`;
* `REMINDER_TOOLCALL_MAX_ROUNDS` — лимит циклов agentic loop
  («LLM → tool_calls → результат → LLM»), общий для всех MCP-серверов
  (по умолчанию 10 итераций).

Клиентская сторона:

* `src/reminder_client.py` — HTTP MCP-клиент (`DEFAULT_URL =
  http://127.0.0.1:8000/mcp`), используется и для других HTTP-серверов
  (geo, weather, news, CRM);
* `src/mcp_worker.py` — фоновые потоки: `MultiServerListWorker` опрашивает
  **все** серверы (stdio + HTTP) и рисует группы в панели MCP,
  `MCPRemoteCallWorker` вызывает инструмент на HTTP-сервере;
* `src/mcp_panel.py` / `src/mcp_ui.py` — панель инструментов в UI.

---

## 2. MCP-серверы со stdio-транспортом

Запускаются как подпроцесс через Python SDK `mcp`
(`src/mcp_client.py`), обмен — stdin/stdout. Команда задаётся переменной
`MCP_SERVER_COMMAND` (по умолчанию `npx -y open-internet-mcp`), таймаут —
`MCP_TIMEOUT` (30 с), журнал — `MCP_LOG_PATH` (`mcp.log`). Одна операция =
одна сессия: старт → `initialize` → запрос → завершение.

| Сервер | Команда | Назначение |
|---|---|---|
| Собственный Git-сервер | `python -u my_git_server.py` | `get_git_log`, `get_git_status`, `get_git_branches` над локальным git (Node.js не нужен); используется командой `/help` и `app/help_pipeline.py` |
| open-internet-mcp | `npx -y open-internet-mcp` (вариант по умолчанию) | 26 инструментов открытых API (`wikipedia_summary`, `github_search`, `arxiv_search`, `open_meteo_forecast`, `usgs_quakes` и др.), требуется Node.js |
| Тестовый echo-сервер | `python -u tests/fixtures/mcp_echo_server.py` | только тесты/скриншоты панели (`scripts/mcp_ui_screenshot.py`) |

Для `get_git_*` репозиторий по умолчанию — `MCP_GIT_REPO_PATH`
(пусто → корень проекта).

---

## 3. Локальный сервис Ollama

| Параметр | Значение |
|---|---|
| Базовый URL | `http://localhost:11434` (`OLLAMA_BASE_URL`) |
| Нативный API | `POST /api/chat`, `GET /api/tags` (клиент `src/tg_ollama_client.py` для Telegram-бота) |
| OpenAI-совместимый API | `http://localhost:11434/v1` (провайдер в `src/llm_providers.py`) |
| Эмбеддинги RAG | модель `nomic-embed-text` (`OLLAMA_EMBED_MODEL`), считаются локально |
| Запуск | `ollama serve`; launcher — `src/ollama_launcher.py` |
| Таймаут генерации | read = 300 с (холодная загрузка большой модели) |

Наружу Ollama не публикуется: удалённый доступ идёт через отдельный
прокси (см. §6).

---

## 4. Внешние API (исходящие запросы)

| Сервис | Адрес | Где используется |
|---|---|---|
| DeepSeek API | `https://api.deepseek.com` (`DEEPSEEK_BASE_URL`) | основной чат (`src/agent.py`, `app/agent.py`, `app/chat_handler.py`, `src/llm_providers.py`); также сводка новостей в `news_server.py` (OpenAI SDK) |
| Telegram Bot API | через `aiogram` (long polling `getUpdates`, только исходящие соединения) | `telegram_bot.py`, токен — `TELEGRAM_BOT_TOKEN` |
| GitHub API | `https://api.github.com` | `src/pr_watcher.py` + `run_pr_watcher.py` (polling открытых PR, `GITHUB_REPO`, `GITHUB_TOKEN`) |
| ip-api.com | геолокация по IP (без ключа, лимит ~45 запросов/мин) | `geo_server.py` → инструмент `get_user_city` |
| Open-Meteo | `https://geocoding-api.open-meteo.com/v1/search`, `https://api.open-meteo.com/v1/forecast` | `weather_server.py` → инструмент `get_weather` (геокодинг + прогноз) |
| RSS-ленты новостей | `https://news.google.com/rss/search?q={topic}`, `https://www.cnews.ru/inc/rss/news.xml`, `https://3dnews.ru/news/rss/`, `https://habr.com/ru/rss/news/?fl=ru`, `https://www.kommersant.ru/RSS/news.xml`, `https://lenta.ru/rss/news` | `news_server.py` (переопределяются через `NEWS_RSS_FEEDS`) |
| DDNS-провайдеры | `https://www.duckdns.org/update`, `https://api.apertodns.com/update`, `https://dynupdate.no-ip.com/nic/update` | `src/ddns_client.py` (обновление IP для удалённого доступа; запускается как служба/планировщик, не из приложения) |

---

## 5. Локальные хранилища (для ориентира)

Не серверы, но часть контура: SQLite-файлы `data/chat_history.db`,
`data/tasks.db`, `data/long_memory.db`, `data/app_state.db`,
`data/telegram_bot.db`, `data/reviewed_prs.db`, `reminders.db`,
векторное хранилище Chroma (`app/data/chroma_db/chroma.sqlite3`).

---

## 6. Удалённый доступ (по желанию)

`src/remote_access.py` + `docs/remote_access.md`: DDNS-домен → Nginx
(порт UI, по умолчанию 8080, **не** 11434) → прокси на
`http://127.0.0.1:11434` с `proxy_buffering off` и таймаутами ~300 с
(стриминг), доступ по API Key. Nginx и сертификаты — внешние
компоненты, ставит пользователь.

---

## 7. Сводная таблица портов

| Порт | Что слушает |
|---|---|
| 8000 | `reminder_server.py` (MCP HTTP) |
| 8001 | `news_server.py` (MCP HTTP) |
| 8002 | `geo_server.py` (MCP HTTP) |
| 8003 | `weather_server.py` (MCP HTTP) |
| 8004 | `crm_server.py` (MCP HTTP) |
| 8005 | `fs_server.py` (MCP HTTP) |
| 11434 | Ollama (нативный API + `/v1`) |
| 8080 | Nginx-прокси удалённого доступа (если настроен) |
