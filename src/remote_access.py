"""Удалённый доступ к Ollama через Nginx-прокси (ТЗ ai_1_30).

Ollama остаётся на 127.0.0.1:11434 (наружу не открывается!). Из
интернета/сети к ней ведёт одна из трёх схем (настройка ``scheme``):

* classic: роутер (проброс 443/80) → Nginx на ПК (API Key + rate
  limiting + HTTPS) → Ollama; домен DDNS обновляет внешний клиент
  (src/ddns_client.py);
* tunnel (рекомендуется для доступа из интернета): edge Cloudflare →
  cloudflared (исходящее соединение с ПК, БЕЗ проброса портов и DDNS)
  → Nginx на 127.0.0.1 (API Key + rate limit, HTTP) → Ollama;
* lan (локальная сеть, без домена): устройства вашей сети → Nginx на
  0.0.0.0 (API Key + rate limit, HTTP, без сертификатов) → Ollama;
  порт наружу не пробрасывать, правило брандмауэра — только частный
  профиль.

Модуль собирает всё, что нужно для всех схем на стороне приложения:

* :class:`RemoteAccessSettings` — глобальные настройки (URL прокси,
  API Key, режим туннеля, DDNS-провайдер/домен/токен, лимиты, пути
  сертификатов) в таблице ``app_state`` (ключ
  ``ollama_remote_access``); хранилище передаётся клиентом (протокол
  ``get``/``set``), как в OptimizationSettings (ТЗ ai_1_28);
* :func:`generate_api_key` — ключ для Nginx (``secrets.token_hex(16)``,
  НЕ хардкодится и не придумывается вручную);
* :func:`build_nginx_conf` / :func:`write_nginx_conf` — готовый
  nginx.conf: Bearer-авторизация, rate limit (по умолчанию 10 r/s с
  burst 20), HTTPS либо HTTP на 127.0.0.1 в туннеле, стриминг
  (proxy_buffering off);
* :func:`check_proxy` — проверка цепочки «приложение → прокси →
  Ollama» запросом /api/tags с ключом.

Провайдер (src.llm_providers.OllamaProvider) применяет настройки так:
заданы И URL прокси, И API Key — запросы идут через прокси с заголовком
Authorization; иначе — локальный режим без изменений.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import httpx

from .utils import get_logger

logger = get_logger("deepseek-chat.remote-access")

# Ключ в таблице app_state (глобальные настройки, вне профилей)
REMOTE_ACCESS_KEY = "ollama_remote_access"

DEFAULT_OLLAMA_HOST = "127.0.0.1:11434"
DEFAULT_RATE_LIMIT = 10  # запросов в секунду с одного IP
DEFAULT_BURST = 20       # размер очереди сверх rate limit
DEFAULT_INTERVAL_MINUTES = 5
DEFAULT_LISTEN_PORT = 8080  # порт Nginx в схемах Tunnel/LAN
CHECK_TIMEOUT = 10.0

# Схемы доступа (ai_1_30 + lan): домен/проброс/сертификаты нужны по-разному
SCHEME_CLASSIC = "classic"  # DDNS + проброс 443/80 + HTTPS (Let's Encrypt)
SCHEME_TUNNEL = "tunnel"    # Cloudflare Tunnel: без DDNS, портов и сертов
SCHEME_LAN = "lan"          # локальная сеть: HTTP на 0.0.0.0, без домена
SCHEMES = (SCHEME_CLASSIC, SCHEME_TUNNEL, SCHEME_LAN)
SCHEME_LABELS = {
    SCHEME_CLASSIC: "Классическая (DDNS + проброс портов)",
    SCHEME_TUNNEL: "Cloudflare Tunnel (без DDNS и проброса)",
    SCHEME_LAN: "Локальная сеть (LAN, без домена)",
}

# Куда панель UI кладёт сгенерированный nginx.conf (внутри проекта,
# как и прочие артефакты каталога output/)
OUTPUT_SUBDIR = "remote_access"


def default_output_dir() -> Path:
    """Каталог для сгенерированного nginx.conf (output/remote_access)."""
    return Path(__file__).resolve().parents[1] / "output" / OUTPUT_SUBDIR


def generate_api_key() -> str:
    """Криптостойкий API Key для Nginx (16 байт → 32 hex-символа)."""
    return secrets.token_hex(16)


def normalize_host(value: str) -> str:
    """Хост для server_name/сертификатов: без схемы, пути и слешей."""
    clean = str(value or "").strip()
    if "://" in clean:
        clean = urlparse(clean).netloc or ""
    return clean.strip("/").rstrip("/")


class RemoteAccessSettings:
    """Настройки удалённого доступа (глобальные, ключ ollama_remote_access).

    Attributes:
        proxy_url: база прокси (``https://myai.duckdns.org`` или
            ``http://192.168.1.50:8080``); пусто — локальный режим.
        api_key: ключ, который Nginx ждёт в ``Authorization: Bearer …``.
        scheme: схема доступа — ``classic`` (DDNS + проброс 443/80 +
            HTTPS), ``tunnel`` (Cloudflare Tunnel: Nginx на 127.0.0.1,
            TLS на edge CF, без DDNS/сертов/проброса), ``lan``
            (локальная сеть: Nginx на 0.0.0.0 по HTTP, без домена).
        listen_port: порт Nginx в схемах tunnel/lan (1024–65535).
        use_api_key: проверять ли ключ в Nginx; отключать разумно
            только в LAN (доверенная сеть) — тогда браузерный чат
            открывается без ключа; вне LAN панель и генератор ключ
            требуют независимо от этого флага.
        ddns_provider: DDNS-сервис (duckdns / aperto / noip).
        ddns_domain: DDNS-домен (``myai.duckdns.org``).
        ddns_token: токен/ключ DDNS-сервиса (у No-IP — логин).
        ddns_secret: пароль No-IP (остальным не нужен).
        ddns_interval_minutes: интервал обновления IP (1–1440).
        rate_limit: Nginx rate limit, запросов в секунду с одного IP.
        burst: размер очереди Nginx сверх rate limit.
        cert_path: fullchain.pem (Let's Encrypt через Certbot).
        cert_key_path: privkey.pem.
        ollama_host: адрес Ollama за Nginx (127.0.0.1:11434).
    """

    DEFAULTS: Dict[str, Any] = {
        "proxy_url": "",
        "api_key": "",
        "scheme": SCHEME_CLASSIC,
        "listen_port": DEFAULT_LISTEN_PORT,
        "use_api_key": True,
        "ddns_provider": "duckdns",
        "ddns_domain": "",
        "ddns_token": "",
        "ddns_secret": "",
        "ddns_interval_minutes": DEFAULT_INTERVAL_MINUTES,
        "rate_limit": DEFAULT_RATE_LIMIT,
        "burst": DEFAULT_BURST,
        "cert_path": "",
        "cert_key_path": "",
        "ollama_host": DEFAULT_OLLAMA_HOST,
    }

    # (ключ провайдера, подпись для UI)
    DDNS_PROVIDERS: Tuple[Tuple[str, str], ...] = (
        ("duckdns", "DuckDNS"),
        ("aperto", "ApertoDNS"),
        ("noip", "No-IP"),
    )

    # Допустимые диапазоны числовых полей (обрезание, а не ошибка)
    LIMITS: Dict[str, Tuple[int, int]] = {
        "ddns_interval_minutes": (1, 1440),
        "rate_limit": (1, 1000),
        "burst": (1, 100000),
        "listen_port": (1024, 65535),
    }

    @staticmethod
    def provider_ids() -> Tuple[str, ...]:
        """Ключи DDNS-провайдеров (для валидации и UI)."""
        return tuple(key for key, _ in RemoteAccessSettings.DDNS_PROVIDERS)

    @staticmethod
    def sanitize(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Приводит настройки к валидному виду (типы и диапазоны).

        Неизвестные ключи отбрасываются, кривые числа заменяются
        дефолтами, строки trim'ятся; пустой ollama_host — дефолт.
        Настройки формата ai_1_30-tunnel (tunnel_mode/tunnel_port)
        мигрируют в scheme/listen_port.
        """
        settings = RemoteAccessSettings.DEFAULTS.copy()
        if not isinstance(raw, dict):
            return settings
        raw_data = dict(raw)
        # Миграция старого формата (ai_1_30-tunnel)
        if "scheme" not in raw_data and raw_data.get("tunnel_mode"):
            raw_data["scheme"] = SCHEME_TUNNEL
        if "listen_port" not in raw_data and "tunnel_port" in raw_data:
            raw_data["listen_port"] = raw_data["tunnel_port"]
        settings["proxy_url"] = str(
            raw_data.get("proxy_url") or "").strip().rstrip("/")
        settings["api_key"] = str(raw_data.get("api_key") or "").strip()
        scheme = str(raw_data.get("scheme") or "").strip().lower()
        if scheme in SCHEMES:
            settings["scheme"] = scheme
        # Отключить ключ можно только осознанно (LAN + доверенная сеть);
        # вне LAN генератор и панель ключ всё равно требуют
        settings["use_api_key"] = bool(raw_data.get("use_api_key", True))
        settings["ddns_domain"] = str(
            raw_data.get("ddns_domain") or "").strip()
        settings["ddns_token"] = str(raw_data.get("ddns_token") or "").strip()
        settings["ddns_secret"] = str(
            raw_data.get("ddns_secret") or "").strip()
        provider = str(raw_data.get("ddns_provider") or "").strip().lower()
        if provider in RemoteAccessSettings.provider_ids():
            settings["ddns_provider"] = provider
        settings["cert_path"] = str(raw_data.get("cert_path") or "").strip()
        settings["cert_key_path"] = str(
            raw_data.get("cert_key_path") or "").strip()
        host = str(raw_data.get("ollama_host") or "").strip()
        settings["ollama_host"] = host or DEFAULT_OLLAMA_HOST
        for key, (low, high) in RemoteAccessSettings.LIMITS.items():
            try:
                value = int(float(raw_data.get(key, settings[key])))
            except (TypeError, ValueError):
                continue  # кривое значение — остаётся дефолт
            settings[key] = min(max(value, low), high)
        return settings

    @staticmethod
    def load(store: Any) -> Dict[str, Any]:
        """Загружает настройки из app_state (или дефолт)."""
        if store is None:
            return RemoteAccessSettings.DEFAULTS.copy()
        try:
            raw = store.get(REMOTE_ACCESS_KEY)
        except Exception as exc:  # noqa: BLE001 — хранилище не критично
            logger.warning("Удалённый доступ: чтение не удалось: %s", exc)
            return RemoteAccessSettings.DEFAULTS.copy()
        if not raw:
            return RemoteAccessSettings.DEFAULTS.copy()
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Удалённый доступ: кривой JSON в %s — откат к дефолтам",
                REMOTE_ACCESS_KEY)
            return RemoteAccessSettings.DEFAULTS.copy()
        return RemoteAccessSettings.sanitize(data)

    @staticmethod
    def save(store: Any, settings: Dict[str, Any]) -> bool:
        """Сохраняет настройки в app_state (upsert, JSON)."""
        if store is None:
            return False
        clean = RemoteAccessSettings.sanitize(settings)
        try:
            return bool(store.set(
                REMOTE_ACCESS_KEY,
                json.dumps(clean, ensure_ascii=False),
            ))
        except Exception as exc:  # noqa: BLE001 — хранилище не критично
            logger.warning("Удалённый доступ: запись не удалась: %s", exc)
            return False


def proxy_active(settings: Optional[Dict[str, Any]]) -> bool:
    """Активен ли прокси-режим для клиента.

    Обычно нужны И URL, И API Key (URL без ключа давал бы 401 на
    каждый запрос). Исключение — сервер без авторизации
    (``use_api_key=False``, разумно только для LAN): тогда достаточно
    URL.
    """
    clean = RemoteAccessSettings.sanitize(settings)
    if not clean["proxy_url"]:
        return False
    return bool(clean["api_key"]) or not clean["use_api_key"]


def proxy_api_base(url: Any) -> str:
    """База прокси без хвоста ``/v1`` и слешей (``""`` — не задан).

    Нативные эндпоинты Ollama (/api/tags, /api/show) и чат живут на
    одной базе: чату OpenAI-SDK нужен суффикс ``/v1``, нативному API —
    нет, поэтому храним базу и добавляем суффикс точечно.
    """
    clean = str(url or "").strip().rstrip("/")
    if clean.endswith("/v1"):
        clean = clean[:-3]
    return clean


def proxy_chat_base(url: Any) -> str:
    """OpenAI-совместимый эндпоинт через прокси (``…/v1``, без дублей)."""
    return proxy_api_base(url) + "/v1"


def default_cert_paths(domain: str) -> Tuple[str, str]:
    """Стандартные пути Certbot на Windows (live/<домен>/…)."""
    clean = normalize_host(domain) or "example.org"
    return (f"C:/Certbot/live/{clean}/fullchain.pem",
            f"C:/Certbot/live/{clean}/privkey.pem")


def build_nginx_conf(domain: str, api_key: str,
                     cert_path: str, cert_key_path: str,
                     ollama_host: str = DEFAULT_OLLAMA_HOST,
                     rate_limit: int = DEFAULT_RATE_LIMIT,
                     burst: int = DEFAULT_BURST,
                     scheme: str = SCHEME_CLASSIC,
                     listen_port: int = DEFAULT_LISTEN_PORT,
                     use_api_key: bool = True,
                     web_root: str = "") -> str:
    """Готовый nginx.conf: rate limit + стриминг (+ API Key auth).

    Три схемы (ТЗ ai_1_30 + lan):

    * ``classic``: 80-й порт перенаправляет на HTTPS (нужен и Certbot'у),
      443 — рабочий с сертификатами; проброс 443/80 на роутере
      обязателен; доступ из интернета; API Key обязателен;
    * ``tunnel`` (Cloudflare Tunnel): Nginx слушает ТОЛЬКО
      ``127.0.0.1:listen_port`` по HTTP (TLS терминируется на edge
      Cloudflare, до Nginx долетает только локальный cloudflared).
      Проброс портов, DDNS и сертификаты не нужны вовсе; rate limit
      считает по настоящему IP клиента из ``CF-Connecting-IP``;
      API Key обязателен;
    * ``lan`` (локальная сеть): Nginx слушает ``0.0.0.0:listen_port``
      по HTTP — доступ только из своей сети (порт наружу НЕ
      пробрасывать!). Настоящие IP устройств видны напрямую, лимит —
      по ``$binary_remote_addr``; домен и сертификаты не нужны;
      API Key по умолчанию включён, но для доверенной сети его можно
      отключить (``use_api_key=False``) — тогда по ``/`` отдаётся
      страница веб-чата из ``web_root`` (chat.html, генерируется
      отдельно) и браузер подключается без ключа.

    Args:
        domain: домен (server_name; в LAN не используется — ``_``).
        api_key: ключ из :func:`generate_api_key` (НЕ придумывать
            вручную); в LAN без ключа не используется.
        cert_path: fullchain.pem (в схемах tunnel/lan не используется).
        cert_key_path: privkey.pem (в схемах tunnel/lan не используется).
        ollama_host: адрес Ollama (127.0.0.1:11434 — НЕ 0.0.0.0!).
        rate_limit: запросов в секунду с одного IP.
        burst: очередь сверх лимита (nodelay).
        scheme: схема доступа (classic / tunnel / lan).
        listen_port: порт Nginx в схемах tunnel/lan.
        use_api_key: проверять ключ (отключение учитывается только в LAN).
        web_root: каталог со страницей чата (chat.html) для LAN;
            пусто — конфиг без веб-чата (только API).

    Returns:
        Текст конфигурации (в UI — записывается через
        :func:`write_nginx_conf`).
    """
    if scheme == SCHEME_TUNNEL:
        return _build_tunnel_conf(
            normalize_host(domain) or "example.org", api_key,
            ollama_host or DEFAULT_OLLAMA_HOST,
            rate_limit, burst, listen_port)
    if scheme == SCHEME_LAN:
        return _build_lan_conf(
            api_key, ollama_host or DEFAULT_OLLAMA_HOST,
            rate_limit, burst, listen_port,
            use_api_key=use_api_key, web_root=web_root)
    # Вне LAN ключ обязателен независимо от флага (доступ из интернета)
    return _build_classic_conf(
        normalize_host(domain) or "example.org", api_key,
        cert_path, cert_key_path,
        ollama_host or DEFAULT_OLLAMA_HOST, rate_limit, burst)


def _build_classic_conf(host: str, api_key: str, cert_path: str,
                        cert_key_path: str, upstream: str,
                        rate_limit: int, burst: int) -> str:
    """Классическая схема: DDNS + проброс портов + Let's Encrypt."""
    return f"""# nginx.conf — удалённый доступ к локальной Ollama ({host})
# Сгенерировано приложением (ТЗ ai_1_30). Ключ НЕ хардкодьте в коде:
# он лежит только здесь и в настройках клиентов.

# Rate limiting: {rate_limit} запросов/с с одного IP (зона 10 МБ)
limit_req_zone $binary_remote_addr zone=ollama_limit:10m rate={int(rate_limit)}r/s;

upstream ollama_backend {{
    server {upstream};
    keepalive 32;
}}

server {{
    listen 80;
    server_name {host};
    # Только перенаправление на HTTPS (и endpoint для Certbot)
    return 301 https://$host$request_uri;
}}

server {{
    listen 443 ssl http2;
    server_name {host};

    # HTTPS (Let's Encrypt, обновление — certbot renew)
    ssl_certificate     {cert_path};
    ssl_certificate_key {cert_key_path};
    ssl_protocols TLSv1.2 TLSv1.3;

    # API Key auth: без заголовка «Authorization: Bearer <ключ>» — 401
    set $api_key "Bearer {api_key}";
    if ($http_authorization != $api_key) {{
        return 401 '{{"error": "Unauthorized"}}';
    }}

    # Rate limiting: сверх {rate_limit} r/s (+burst {burst}) — 429
    limit_req zone=ollama_limit burst={int(burst)} nodelay;
    limit_req_status 429;

    location / {{
        proxy_pass http://ollama_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Стриминг ответа: буферизация рвёт поток токенов
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        chunked_transfer_encoding on;
    }}
}}
"""


def _build_tunnel_conf(host: str, api_key: str, upstream: str,
                       rate_limit: int, burst: int,
                       tunnel_port: int) -> str:
    """Схема Cloudflare Tunnel: HTTP на 127.0.0.1, TLS — на edge CF.

    За туннелем Nginx видит все соединения с одного адреса (127.0.0.1 —
    cloudflared), поэтому лимит считается по заголовку
    ``CF-Connecting-IP``: его выставляет edge Cloudflare и подделать
    через туннель нельзя (единственный путь к listener'у — cloudflared,
    ведь порт слушает только 127.0.0.1). Пустой заголовок — локальный
    запрос с этой же машины: ключом становится remote_addr.
    """
    return f"""# nginx.conf — Ollama через Cloudflare Tunnel ({host})
# Сгенерировано приложением (ТЗ ai_1_30, вариант C). Nginx слушает
# ТОЛЬКО 127.0.0.1:{int(tunnel_port)} — наружу не смотрит, проброс
# портов не нужен: до него долетает только cloudflared этой машины.

map $http_cf_connecting_ip $rate_limit_key {{
    ""      $binary_remote_addr;
    default $http_cf_connecting_ip;
}}

# Rate limiting: {rate_limit} запросов/с на настоящего клиента
# (зона 10 МБ; IP клиента — из CF-Connecting-IP, см. map выше)
limit_req_zone $rate_limit_key zone=ollama_limit:10m rate={int(rate_limit)}r/s;

upstream ollama_backend {{
    server {upstream};
    keepalive 32;
}}

# TLS терминируется на edge Cloudflare — локально достаточно HTTP
server {{
    listen 127.0.0.1:{int(tunnel_port)};
    server_name {host};

    # API Key auth: без заголовка «Authorization: Bearer <ключ>» — 401
    set $api_key "Bearer {api_key}";
    if ($http_authorization != $api_key) {{
        return 401 '{{"error": "Unauthorized"}}';
    }}

    # Rate limiting: сверх {rate_limit} r/s (+burst {burst}) — 429
    limit_req zone=ollama_limit burst={int(burst)} nodelay;
    limit_req_status 429;

    location / {{
        proxy_pass http://ollama_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;

        # Стриминг ответа: буферизация рвёт поток токенов
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        chunked_transfer_encoding on;
    }}
}}
"""


def _build_lan_conf(api_key: str, upstream: str,
                    rate_limit: int, burst: int, listen_port: int,
                    use_api_key: bool = True,
                    web_root: str = "") -> str:
    """Схема LAN: HTTP на всех интерфейсах, доступ только из своей сети.

    В отличие от туннеля, Nginx видит настоящий IP каждого устройства —
    лимит считается по ``$binary_remote_addr``. Домен не нужен
    (``server_name _`` — любой Host). По желанию (``use_api_key=False``)
    проверка ключа отключается, а ``web_root`` подключает страницу
    веб-чата по ``/`` — тогда с телефона достаточно открыть адрес в
    браузере. Правило брандмауэра Windows — только для частного
    профиля, порт на роутере не пробрасывать.
    """
    auth_block = ""
    if use_api_key:
        auth_block = f"""
    # API Key auth: без заголовка «Authorization: Bearer <ключ>» — 401
    set $api_key "Bearer {api_key}";
    if ($http_authorization != $api_key) {{
        return 401 '{{"error": "Unauthorized"}}';
    }}
"""
    web_block = ""
    if web_root:
        web_block = f"""
    # Веб-чат: страница {web_root}/chat.html открывается по /
    location = / {{
        root {web_root};
        try_files /chat.html =404;
    }}
"""
    key_note = ("API Key обязателен (заголовок «Authorization: Bearer»)"
                if use_api_key else "БЕЗ API Key — только доверенная сеть")
    return f"""# nginx.conf — Ollama для локальной сети (Nginx → {upstream})
# Сгенерировано приложением (ТЗ ai_1_30, схема LAN; {key_note}).
# Nginx слушает 0.0.0.0:{int(listen_port)} — доступ ТОЛЬКО из вашей
# сети: не пробрасывайте этот порт на роутере и держите правило
# брандмауэра Windows только для частного профиля. Ollama остаётся
# на 127.0.0.1.

# Rate limiting: {rate_limit} запросов/с с одного устройства
# (в LAN Nginx видит настоящий IP каждого клиента)
limit_req_zone $binary_remote_addr zone=ollama_limit:10m rate={int(rate_limit)}r/s;

upstream ollama_backend {{
    server {upstream};
    keepalive 32;
}}

# HTTP без TLS: трафик не покидает вашу сеть — сертификаты не нужны
server {{
    listen 0.0.0.0:{int(listen_port)};
    server_name _;
{auth_block}{web_block}
    # Rate limiting: сверх {rate_limit} r/s (+burst {burst}) — 429
    limit_req zone=ollama_limit burst={int(burst)} nodelay;
    limit_req_status 429;

    # API Ollama: /api/... (нативный) и /v1/... (OpenAI-совместимый)
    location / {{
        proxy_pass http://ollama_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Стриминг ответа: буферизация рвёт поток токенов
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        chunked_transfer_encoding on;
    }}
}}
"""


def write_nginx_conf(out_path: Any, **kwargs: Any) -> Path:
    """Генерирует nginx.conf и записывает его (каталог создаётся).

    Args:
        out_path: путь файла (строка/Path).
        **kwargs: аргументы :func:`build_nginx_conf`.

    Returns:
        Path записанного файла.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_nginx_conf(**kwargs), encoding="utf-8")
    logger.info("nginx.conf записан: %s", path)
    return path


def check_proxy(proxy_url: str, api_key: str,
                timeout: float = CHECK_TIMEOUT) -> Tuple[bool, str]:
    """Проверяет цепочку «приложение → Nginx → Ollama» запросом /api/tags.

    Args:
        proxy_url: база прокси (https://myai.duckdns.org).
        api_key: ключ для заголовка Authorization.

    Returns:
        (успех, сообщение: число моделей / 401 / ошибка сети).
    """
    base = proxy_api_base(proxy_url)
    if not base:
        return False, "URL прокси не задан"
    # Без ключа заголовок не отправляем (сервер LAN без API Key отвечает 200)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        response = httpx.get(
            f"{base}/api/tags",
            headers=headers,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 — сеть/DNS/TLS
        return False, f"Прокси недоступен: {exc}"
    if response.status_code == 200:
        try:
            models = len(response.json().get("models") or [])
        except ValueError:
            models = 0
        return True, f"Соединение OK — моделей в Ollama: {models}"
    if response.status_code == 401:
        return False, "401 Unauthorized — Nginx отклонил API Key"
    return False, f"Неожиданный ответ прокси: HTTP {response.status_code}"
