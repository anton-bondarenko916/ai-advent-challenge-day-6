"""Страница веб-чата для доступа к Ollama с телефона/ноутбука (ai_1_30).

Один самодостаточный HTML-файл (без зависимостей и сборки), который
Nginx в схеме LAN раздаёт по адресу ``http://<IP-ПК>:<порт>/``.
Сценарий: открыли адрес в браузере — общаетесь с моделью.

Как работает:

* список моделей — GET ``/api/tags`` (нативный API Ollama за Nginx);
* отправка — POST ``/api/chat`` со стримингом (``stream: true``):
  ответ приходит NDJSON-потоком, текст дописывается по мере генерации;
* история диалога живёт ТОЛЬКО в памяти открытой вкладки (кнопка
  «Очистить» или закрытие вкладки её сбрасывает);
* если сервер требует API Key (классический LAN-конфиг с auth), при
  первом 401 страница спрашивает ключ и хранит его в localStorage —
  дальше подставляет заголовок Authorization сама;
* ответ модели вставляется через ``textContent`` (никакого innerHTML
  для данных модели — защита от XSS через ответ модели);
* перед отправкой история урезается до последних 20 сообщений —
  контекст Ollama не бесконечный.

Раздаётся конфигом из :func:`src.remote_access.build_nginx_conf`
(location ``= /`` с ``try_files /chat.html``), запись —
:func:`write_chat_page`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

CHAT_PAGE_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ollama — чат (LAN)</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: system-ui, -apple-system, sans-serif;
    display: flex; flex-direction: column; height: 100dvh;
    background: #f4f6f9; color: #1d2129;
  }
  header {
    display: flex; gap: 8px; align-items: center;
    padding: 8px 12px; background: #fff;
    border-bottom: 1px solid #d9dfe7;
  }
  header select { flex: 1; padding: 6px; font-size: 14px; }
  header button { padding: 6px 10px; font-size: 14px; }
  #chat {
    flex: 1; overflow-y: auto; padding: 12px;
    display: flex; flex-direction: column; gap: 8px;
  }
  .msg {
    max-width: 85%; padding: 8px 12px; border-radius: 12px;
    white-space: pre-wrap; word-break: break-word; line-height: 1.45;
    font-size: 15px;
  }
  .user { align-self: flex-end; background: #0b57d0; color: #fff; }
  .bot  { align-self: flex-start; background: #e6ebf2; }
  .err  { align-self: center; color: #b3261e; font-size: 13px; }
  form {
    display: flex; gap: 8px; padding: 8px 12px;
    background: #fff; border-top: 1px solid #d9dfe7;
  }
  form textarea {
    flex: 1; resize: none; padding: 8px; font: inherit;
    min-height: 40px; max-height: 140px;
  }
  form button { padding: 8px 16px; font-size: 15px; }
</style>
</head>
<body>
<header>
  <select id="model" title="Модель"></select>
  <button type="button" id="clear">Очистить</button>
</header>
<div id="chat"></div>
<form id="form">
  <textarea id="input" rows="1" placeholder="Сообщение… (Enter — отправить, Shift+Enter — перенос)"></textarea>
  <button id="send" type="submit">➤</button>
</form>
<script>
"use strict";
const chat = document.getElementById("chat");
const input = document.getElementById("input");
const modelSel = document.getElementById("model");
const sendBtn = document.getElementById("send");
// История диалога: только в памяти вкладки, ничего не пишется на диск
let history = [];
let busy = false;

function addMsg(role, text) {
  const div = document.createElement("div");
  div.className = "msg " + (role === "user" ? "user" : "bot");
  // ТОЛЬКО textContent: ответ модели не исполняется как HTML
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function authHeaders() {
  const key = localStorage.getItem("ollama_key");
  return key ? { "Authorization": "Bearer " + key } : {};
}

function askKey() {
  const key = prompt("Сервер требует API Key:");
  if (key) {
    localStorage.setItem("ollama_key", key.trim());
    return true;
  }
  return false;
}

async function loadModels() {
  try {
    const resp = await fetch("/api/tags", { headers: authHeaders() });
    if (resp.status === 401) {
      if (askKey()) { loadModels(); return; }
      addMsg("err", "401: без API Key сервер не отвечает");
      return;
    }
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    modelSel.textContent = "";
    for (const m of data.models || []) {
      const opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = m.name;
      modelSel.appendChild(opt);
    }
  } catch (e) {
    addMsg("err", "Сервер недоступен: " + e.message);
  }
}

async function send() {
  const text = input.value.trim();
  if (!text || busy) return;
  busy = true;
  sendBtn.disabled = true;
  input.value = "";
  addMsg("user", text);
  history.push({ role: "user", content: text });
  const bot = addMsg("bot", "…");
  let acc = "";
  try {
    // Контекст модели ограничен: отправляем последние 20 сообщений
    const payload = history.length > 20 ? history.slice(-20) : history;
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: Object.assign(
        { "Content-Type": "application/json" }, authHeaders()),
      body: JSON.stringify({
        model: modelSel.value, messages: payload, stream: true
      })
    });
    if (resp.status === 401) {
      askKey();
      throw new Error("401 — нужен API Key (введите и повторите)");
    }
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const part = await reader.read();
      if (part.done) break;
      buf += decoder.decode(part.value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        const chunk = JSON.parse(line);
        if (chunk.error) throw new Error(chunk.error);
        if (chunk.message && chunk.message.content) {
          acc += chunk.message.content;
          bot.textContent = acc;
          chat.scrollTop = chat.scrollHeight;
        }
      }
    }
    if (acc) {
      history.push({ role: "assistant", content: acc });
    } else {
      bot.textContent = "(пустой ответ)";
    }
  } catch (e) {
    bot.textContent = "Ошибка: " + e.message;
    bot.className = "msg err";
  } finally {
    busy = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

document.getElementById("form").addEventListener("submit", (e) => {
  e.preventDefault();
  send();
});
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
document.getElementById("clear").addEventListener("click", () => {
  history = [];
  chat.textContent = "";
  input.focus();
});

loadModels();
input.focus();
</script>
</body>
</html>
"""


def write_chat_page(out_path: Union[str, Path]) -> Path:
    """Записывает страницу чата (каталог создаётся при необходимости).

    Args:
        out_path: путь файла (обычно ``<output>/remote_access/chat.html``).

    Returns:
        Path записанного файла.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CHAT_PAGE_HTML, encoding="utf-8")
    return path
