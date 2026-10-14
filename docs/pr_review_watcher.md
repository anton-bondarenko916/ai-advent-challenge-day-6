# AI-ревью PR: watcher GitHub + RAG по коду + Inspectra

Автономный ревьюер PR хостится на вашем ПК. Цикл: опрос GitHub API
каждые 5 минут (только исходящие соединения) → новый открытый PR →
RAG по коду проекта находит релевантные `*.py`-чанки → промпт «баги и
edge cases» в `.inspectra.yml` → локальная модель `qwen2.5-coder-14b-32k`
через Inspectra CLI → комментарий в PR.

```text
[GitHub PR] ⇄ polling 5 мин ⇄ run_pr_watcher.py (ваш ПК)
                                 ⇅ RAG-поиск (Chroma, коллекция project_code)
                             Ollama qwen2.5-coder-14b-32k (num_ctx 32768)
                                 ⇅ inspectra review --post-comment
                            комментарий «баги и edge cases» в PR
```

Ключ повторного ревью — пара `(pr_number, head_sha)` в SQLite
(`data/reviewed_prs.db`): один и тот же коммит не ревьюится дважды,
новый коммит в PR (новый head_sha) запускает ревью снова.

## Шаг 1. Модель с контекстом 32k

`num_ctx` задаётся **только через Modelfile** — при запуске модели его
переопределить нельзя.

```cmd
cd C:\AI\AI_2_1\deepseek-chat-app
ollama pull qwen2.5-coder:14b
ollama create qwen2.5-coder-14b-32k -f Modelfile_qwen_coder_32k
ollama show qwen2.5-coder-14b-32k
```

`ollama show` в блоке Parameters должен показать `num_ctx 32768`.

## Шаг 2. Inspectra CLI и токен GitHub

1. Установите Inspectra (pip-пакет, исполняемый файл попадёт в `Scripts`
   Python — он уже в PATH):

```cmd
pip install inspectra
inspectra version
```

2. Создайте персональный токен: GitHub → Settings → Developer settings
   → Personal access tokens — с правами **repo** (чтение PR +
   публикация комментариев; без `repo` watcher бесполезен).
3. Впишите в `.env` в корне проекта:

```env
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
GITHUB_REPO=owner/repo            # можно и полную ссылку https://github.com/owner/repo
# Необязательно (значения по умолчанию):
# PR_WATCHER_INTERVAL=5            # минуты между опросами GitHub
# PR_WATCHER_PROJECT_PATH=         # "" — индексировать этот же проект
# PR_WATCHER_MODEL=qwen2.5-coder-14b-32k
# PR_WATCHER_DB_PATH=data/reviewed_prs.db
# PR_WATCHER_TIMEOUT=600           # таймаут одного запуска inspectra, сек
# PR_WATCHER_TOP_K=5               # чанков RAG-контекста в промпт
```

## Шаг 3. Запуск

```cmd
cd C:\AI\AI_2_1\deepseek-chat-app
python run_pr_watcher.py
```

При старте индексируются все `*.py` проекта (кроме `venv`, `.venv`,
`__pycache__`, `.git`) в отдельную коллекцию `project_code` Chroma
(`app/data/chroma_db`); статистика — в журнал. Затем запускается цикл
polling. Журнал — `run_pr_watcher.log` (UTF-8) + консоль. Остановка —
Ctrl+C.

### Автозапуск после перезагрузки ПК

```cmd
schtasks /Create /TN "PR Review Watcher" /SC ONSTART /RU SYSTEM ^
  /TR "cmd /c cd /d C:\AI\AI_2_1\deepseek-chat-app && python run_pr_watcher.py"
```

Рабочий каталог важен: иначе watcher не найдёт `.env`. Удалить задачу:
`schtasks /Delete /TN "PR Review Watcher" /F`.

## Как ревьюируется PR

1. **RAG-контекст**: по заголовку PR ищутся топ-K чанков кода
   (`PR_WATCHER_TOP_K`); каждый чанк — до 500 символов с путём файла.
2. **Промпт** пишется во временный `.inspectra.yml` проекта-ревью:
   инструкции «найди баги, необработанные edge cases, логические
   ошибки в условиях и циклах; для каждой находки — severity и
   рекомендация». Стиль/форматирование явно исключены.
3. **Запуск**: `inspectra review --pr N --provider ollama --model
   qwen2.5-coder-14b-32k --post-comment --pr-summary --config ...`.
   Флаг `--post-comment` обязателен — по умолчанию его НЕТ.
4. **Уборка**: временный конфиг удаляется; если у проекта был свой
   `.inspectra.yml` — он восстанавливается байт в байт.

## Поведение при ошибках (watcher не умирает)

| Ошибка | Поведение |
|---|---|
| GitHub недоступен / 401 / rate limit | запись в лог, повтор на следующем цикле |
| RAG-поиск не удался (Ollama занята) | ревью без контекста (заглушка в промпте) |
| `inspectra` не найден / таймаут / код ≠ 0 | `success=False` + лог, watcher продолжает |
| ошибка обработки одного PR | остальные PR цикла всё равно обрабатываются |
| ошибка БД | методы деградируют (лог), ход продолжается |

PR помечается просмотренным **после попытки ревью даже при ошибке** —
иначе сломавшийся PR ретраился бы каждые 5 минут бесконечно. Повторное
ревью придёт с новым коммитом (новым head_sha); форсировать можно
удалением строки из `data/reviewed_prs.db`.

## Ограничения

* Опрос берёт первую страницу открытых PR (50 шт., сортировка по
  обновлению) — для небольших команд этого достаточно.
* PR обрабатываются последовательно — ревью одного PR может занимать
  до `PR_WATCHER_TIMEOUT` секунд (qwen2.5-coder на 32k контекста
  требователен к памяти; на 14B рассчитывайте на GPU с ≥16 ГБ VRAM
  или на медленный CPU-режим).
* Первый запуск может быть долгим: pull модели ~9 ГБ + индексация кода.
