"""Тесты удалённого доступа к Ollama через Nginx-прокси (ТЗ ai_1_30).

Проверяются офлайн (без сети, Nginx и запущенного Ollama; httpx
замокан):

* RemoteAccessSettings: дефолты, санитизация (типы, диапазоны,
  неизвестный провайдер), сохранение/загрузка в app_state (SQLite),
  proxy_active (URL БЕЗ ключа — не активен);
* генерация API Key и nginx.conf (API Key auth, rate limit, HTTPS,
  стриминг, upstream на 127.0.0.1), запись файла;
* check_proxy: 200/401/5xx/сеть (httpx замокан);
* DDNS-клиент: протоколы DuckDNS/ApertoDNS/No-IP (URL, параметры,
  аутентификация, разбор ответов), DDNSClient.update_once (диспетчер,
  ошибки не выбрасываются), циклы ddns_loop/run_forever, CLI;
* OllamaProvider: прокси-режим (base_url …/v1, Bearer-заголовок,
  api_base для /api), таймаут и заголовки проверок доступности,
  горячее set_remote_access, фабрика (DeepSeek не затрагивается);
* ядро: set_ollama_remote_access (санитизация, горячее применение,
  перенос при switch_provider), свойство-копия;
* UI-панель: применение на blur, генерация ключа, DDNS-кнопка,
  проверка соединения, генерация nginx.conf (в tmp), индикатор режима.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import remote_access  # noqa: E402
from src.agent import DeepSeekAgent  # noqa: E402
from src.app_state import AppStateStore  # noqa: E402
from src.ddns_client import (  # noqa: E402
    DDNSClient,
    duckdns_domain,
    ddns_loop,
    update_aperto_dns,
    update_duckdns,
    update_noip,
)
from src.llm_providers import (  # noqa: E402
    DeepSeekProvider,
    LLMProviderFactory,
    OllamaProvider,
    PROXY_TAGS_TIMEOUT,
)
from src.remote_access import (  # noqa: E402
    REMOTE_ACCESS_KEY,
    RemoteAccessSettings,
    build_nginx_conf,
    check_proxy,
    default_cert_paths,
    generate_api_key,
    normalize_host,
    proxy_active,
    proxy_api_base,
    proxy_chat_base,
    write_nginx_conf,
)

from tests.test_app_agent import make_agent  # noqa: E402
from tests.test_flet_views import make_app  # noqa: E402

PROXY = {
    "proxy_url": "https://myai.duckdns.org",
    "api_key": "a" * 32,
}


class _FakeHTTPResponse:
    """Ответ httpx-заглушка (status_code + json()/text)."""

    def __init__(self, status_code: int = 200, payload: dict = None,
                 text: str = "") -> None:
        self.status_code = int(status_code)
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict:
        return self._payload


class _StubAppState:
    """AppStateStore-заглушка (запоминает set)."""

    def __init__(self) -> None:
        self.saved: dict = {}

    def get(self, key: str):
        return self.saved.get(key)

    def set(self, key: str, value: str) -> bool:
        self.saved[key] = value
        return True


# ----------------------------------------------------------------------
# RemoteAccessSettings: дефолты, санитизация, app_state
# ----------------------------------------------------------------------
class RemoteAccessSettingsTest(unittest.TestCase):
    """Санитизация и хранение настроек в app_state."""

    def test_defaults(self) -> None:
        settings = RemoteAccessSettings.DEFAULTS.copy()
        self.assertEqual(settings["proxy_url"], "")
        self.assertEqual(settings["api_key"], "")
        self.assertEqual(settings["ddns_provider"], "duckdns")
        self.assertEqual(settings["ddns_interval_minutes"], 5)
        self.assertEqual(settings["rate_limit"], 10)
        self.assertEqual(settings["burst"], 20)
        self.assertEqual(settings["ollama_host"], "127.0.0.1:11434")

    def test_sanitize_strips_and_drops_unknown(self) -> None:
        clean = RemoteAccessSettings.sanitize({
            "proxy_url": " https://myai.duckdns.org/// ",
            "api_key": "  key123\n",
            "ddns_provider": "DUCKDNS",
            "ddns_domain": " myai.duckdns.org ",
            "unknown_key": "мусор",
        })
        self.assertEqual(clean["proxy_url"], "https://myai.duckdns.org")
        self.assertEqual(clean["api_key"], "key123")
        self.assertEqual(clean["ddns_provider"], "duckdns")
        self.assertEqual(clean["ddns_domain"], "myai.duckdns.org")
        self.assertNotIn("unknown_key", clean)

    def test_sanitize_unknown_provider_falls_back(self) -> None:
        clean = RemoteAccessSettings.sanitize({"ddns_provider": "gopher"})
        self.assertEqual(clean["ddns_provider"], "duckdns")
        clean = RemoteAccessSettings.sanitize({"ddns_provider": "noip"})
        self.assertEqual(clean["ddns_provider"], "noip")

    def test_sanitize_clamps_numbers(self) -> None:
        clean = RemoteAccessSettings.sanitize({
            "ddns_interval_minutes": 99999,
            "rate_limit": "50",      # строка-число приводится
            "burst": "кривое",       # кривое — остаётся дефолт
        })
        self.assertEqual(clean["ddns_interval_minutes"], 1440)
        self.assertEqual(clean["rate_limit"], 50)
        self.assertEqual(clean["burst"], 20)

    def test_sanitize_empty_ollama_host_defaults(self) -> None:
        clean = RemoteAccessSettings.sanitize({"ollama_host": "   "})
        self.assertEqual(clean["ollama_host"], "127.0.0.1:11434")

    def test_sanitize_scheme_and_listen_port(self) -> None:
        clean = RemoteAccessSettings.sanitize(
            {"scheme": "lan", "listen_port": "9001"})
        self.assertEqual(clean["scheme"], "lan")
        self.assertEqual(clean["listen_port"], 9001)
        # Порт вне диапазона обрезается, неизвестная схема — classic
        self.assertEqual(
            RemoteAccessSettings.sanitize({"listen_port": 80})[
                "listen_port"], 1024)
        self.assertEqual(
            RemoteAccessSettings.sanitize({"scheme": "gopher"})["scheme"],
            "classic")
        defaults = RemoteAccessSettings.sanitize({})
        self.assertEqual(defaults["scheme"], "classic")
        self.assertEqual(defaults["listen_port"], 8080)

    def test_sanitize_migrates_legacy_tunnel_fields(self) -> None:
        """Формат ai_1_30-tunnel (tunnel_mode/tunnel_port) читается."""
        clean = RemoteAccessSettings.sanitize(
            {"tunnel_mode": True, "tunnel_port": 9443})
        self.assertEqual(clean["scheme"], "tunnel")
        self.assertEqual(clean["listen_port"], 9443)
        # Старый формат без туннеля — классика
        clean = RemoteAccessSettings.sanitize({"tunnel_mode": False})
        self.assertEqual(clean["scheme"], "classic")

    def test_sanitize_non_dict_returns_defaults(self) -> None:
        self.assertEqual(
            RemoteAccessSettings.sanitize("мусор"),
            RemoteAccessSettings.DEFAULTS.copy())
        self.assertEqual(
            RemoteAccessSettings.sanitize(None),
            RemoteAccessSettings.DEFAULTS.copy())

    def test_roundtrip_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AppStateStore(Path(tmp) / "app_state.db")
            RemoteAccessSettings.save(store, {
                "proxy_url": "https://myai.duckdns.org",
                "api_key": "k" * 32,
                "ddns_provider": "aperto",
                "ddns_domain": "myai.apertodns.com",
                "rate_limit": 25,
            })
            loaded = RemoteAccessSettings.load(store)
        self.assertEqual(loaded["proxy_url"], "https://myai.duckdns.org")
        self.assertEqual(loaded["api_key"], "k" * 32)
        self.assertEqual(loaded["ddns_provider"], "aperto")
        self.assertEqual(loaded["ddns_domain"], "myai.apertodns.com")
        self.assertEqual(loaded["rate_limit"], 25)
        self.assertEqual(REMOTE_ACCESS_KEY, "ollama_remote_access")

    def test_load_corrupt_json_falls_back(self) -> None:
        store = _StubAppState()
        store.set(REMOTE_ACCESS_KEY, "{не json")
        self.assertEqual(RemoteAccessSettings.load(store),
                         RemoteAccessSettings.DEFAULTS.copy())

    def test_load_without_store(self) -> None:
        self.assertEqual(RemoteAccessSettings.load(None),
                         RemoteAccessSettings.DEFAULTS.copy())

    def test_save_broken_store_returns_false(self) -> None:
        class _Broken:
            def set(self, key, value):
                raise OSError("db locked")

        self.assertFalse(RemoteAccessSettings.save(_Broken(), {}))

    def test_proxy_active_requires_url_and_key(self) -> None:
        self.assertTrue(proxy_active(dict(PROXY)))
        self.assertFalse(proxy_active({"proxy_url": "https://x.org"}))
        self.assertFalse(proxy_active({"api_key": "k"}))
        self.assertFalse(proxy_active(None))

    def test_proxy_active_without_key_when_server_has_no_auth(self) -> None:
        """LAN без API Key (use_api_key=False): достаточно URL."""
        self.assertTrue(proxy_active({
            "proxy_url": "http://192.168.1.50:8080", "use_api_key": False}))
        # Без явного флага — прежнее правило «нужны оба поля»
        self.assertFalse(proxy_active({"proxy_url": "http://192.168.1.50"}))

    def test_sanitize_use_api_key(self) -> None:
        self.assertTrue(RemoteAccessSettings.sanitize({})["use_api_key"])
        self.assertFalse(RemoteAccessSettings.sanitize(
            {"use_api_key": False})["use_api_key"])
        self.assertTrue(RemoteAccessSettings.sanitize(
            {"use_api_key": "да"})["use_api_key"])


# ----------------------------------------------------------------------
# URL прокси и API Key
# ----------------------------------------------------------------------
class ProxyUrlTest(unittest.TestCase):
    """Нормализация базы прокси (/v1 без дублей) и API Key."""

    def test_proxy_api_base_strips_suffixes(self) -> None:
        self.assertEqual(proxy_api_base("https://x.org"),
                         "https://x.org")
        self.assertEqual(proxy_api_base("https://x.org/"),
                         "https://x.org")
        self.assertEqual(proxy_api_base("https://x.org/v1/"),
                         "https://x.org")
        self.assertEqual(proxy_api_base(None), "")

    def test_proxy_chat_base_appends_v1_once(self) -> None:
        self.assertEqual(proxy_chat_base("https://x.org"),
                         "https://x.org/v1")
        self.assertEqual(proxy_chat_base("https://x.org/v1"),
                         "https://x.org/v1")

    def test_normalize_host(self) -> None:
        self.assertEqual(normalize_host("https://x.org/path"), "x.org")
        self.assertEqual(normalize_host("X.ORG/"), "X.ORG")
        self.assertEqual(normalize_host(""), "")

    def test_generate_api_key(self) -> None:
        key = generate_api_key()
        self.assertEqual(len(key), 32)
        self.assertRegex(key, r"^[0-9a-f]{32}$")
        self.assertNotEqual(key, generate_api_key())

    def test_default_cert_paths_contain_domain(self) -> None:
        cert, key = default_cert_paths("myai.duckdns.org")
        self.assertEqual(cert,
                         "C:/Certbot/live/myai.duckdns.org/fullchain.pem")
        self.assertEqual(key,
                         "C:/Certbot/live/myai.duckdns.org/privkey.pem")


# ----------------------------------------------------------------------
# nginx.conf: API Key auth + rate limit + HTTPS + стриминг
# ----------------------------------------------------------------------
class NginxConfTest(unittest.TestCase):
    """Генерация конфига Nginx по ТЗ."""

    def setUp(self) -> None:
        self.conf = build_nginx_conf(
            "myai.duckdns.org", "key123",
            "C:/certs/fullchain.pem", "C:/certs/privkey.pem")

    def test_api_key_auth(self) -> None:
        self.assertIn('set $api_key "Bearer key123";', self.conf)
        self.assertIn("if ($http_authorization != $api_key)", self.conf)
        self.assertIn("return 401", self.conf)

    def test_rate_limiting(self) -> None:
        self.assertIn("rate=10r/s", self.conf)
        self.assertIn("zone=ollama_limit:10m", self.conf)
        self.assertIn("burst=20 nodelay", self.conf)
        self.assertIn("limit_req_status 429", self.conf)

    def test_https(self) -> None:
        self.assertIn("listen 443 ssl http2;", self.conf)
        self.assertIn("ssl_certificate     C:/certs/fullchain.pem;",
                      self.conf)
        self.assertIn("ssl_certificate_key C:/certs/privkey.pem;",
                      self.conf)
        self.assertIn("ssl_protocols TLSv1.2 TLSv1.3;", self.conf)
        # 80-й только перенаправляет на HTTPS
        self.assertIn("listen 80;", self.conf)
        self.assertIn("return 301 https://$host$request_uri;", self.conf)
        self.assertIn("server_name myai.duckdns.org;", self.conf)

    def test_upstream_is_loopback_only(self) -> None:
        self.assertIn("server 127.0.0.1:11434;", self.conf)
        self.assertNotIn("0.0.0.0", self.conf)

    def test_streaming(self) -> None:
        self.assertIn("proxy_pass http://ollama_backend;", self.conf)
        self.assertIn("proxy_buffering off;", self.conf)
        self.assertIn("proxy_read_timeout 300s;", self.conf)
        self.assertIn("chunked_transfer_encoding on;", self.conf)

    def test_custom_limits_and_host(self) -> None:
        conf = build_nginx_conf(
            "x.org", "k", "c.pem", "c.key",
            ollama_host="127.0.0.1:11500", rate_limit=5, burst=7)
        self.assertIn("rate=5r/s", conf)
        self.assertIn("burst=7 nodelay", conf)
        self.assertIn("server 127.0.0.1:11500;", conf)

    def test_write_nginx_conf_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "nginx.conf"
            path = write_nginx_conf(
                out, domain="myai.duckdns.org", api_key="key123",
                cert_path="c.pem", cert_key_path="c.key")
            self.assertEqual(path, out)
            content = out.read_text(encoding="utf-8")
        self.assertIn("server_name myai.duckdns.org;", content)
        self.assertIn("Bearer key123", content)


class NginxTunnelConfTest(unittest.TestCase):
    """Генерация nginx.conf для схемы Cloudflare Tunnel."""

    def setUp(self) -> None:
        self.conf = build_nginx_conf(
            "ai.example.com", "key123", "", "",
            scheme="tunnel", listen_port=8080)

    def test_http_on_loopback_only(self) -> None:
        self.assertIn("listen 127.0.0.1:8080;", self.conf)
        self.assertIn("server_name ai.example.com;", self.conf)
        # TLS — на edge Cloudflare: локально сертификатов и 443 нет
        self.assertNotIn("listen 443", self.conf)
        self.assertNotIn("ssl_certificate", self.conf)
        self.assertNotIn("listen 80;", self.conf)
        self.assertNotIn("return 301", self.conf)

    def test_api_key_and_rate_limit_kept(self) -> None:
        self.assertIn('set $api_key "Bearer key123";', self.conf)
        self.assertIn("if ($http_authorization != $api_key)", self.conf)
        self.assertIn("return 401", self.conf)
        self.assertIn("rate=10r/s", self.conf)
        self.assertIn("burst=20 nodelay", self.conf)
        self.assertIn("limit_req_status 429", self.conf)

    def test_rate_limit_key_is_real_client_ip(self) -> None:
        # За туннелем все соединения с 127.0.0.1 (cloudflared) — лимит
        # считается по CF-Connecting-IP, локальные запросы — по remote_addr
        self.assertIn(
            "map $http_cf_connecting_ip $rate_limit_key", self.conf)
        self.assertIn('""      $binary_remote_addr;', self.conf)
        self.assertIn(
            "limit_req_zone $rate_limit_key zone=ollama_limit:10m",
            self.conf)

    def test_streaming_and_loopback_upstream(self) -> None:
        self.assertIn("server 127.0.0.1:11434;", self.conf)
        self.assertNotIn("0.0.0.0", self.conf)
        self.assertIn("proxy_buffering off;", self.conf)
        self.assertIn("proxy_read_timeout 300s;", self.conf)
        self.assertIn("chunked_transfer_encoding on;", self.conf)

    def test_custom_port(self) -> None:
        conf = build_nginx_conf("ai.example.com", "k", "", "",
                                scheme="tunnel", listen_port=9443)
        self.assertIn("listen 127.0.0.1:9443;", conf)


class NginxLanConfTest(unittest.TestCase):
    """Генерация nginx.conf для схемы локальной сети (LAN)."""

    def setUp(self) -> None:
        self.conf = build_nginx_conf(
            "_", "key123", "", "",
            scheme="lan", listen_port=8080)

    def test_http_on_all_interfaces(self) -> None:
        self.assertIn("listen 0.0.0.0:8080;", self.conf)
        self.assertIn("server_name _;", self.conf)
        # Домен и сертификаты в LAN не нужны
        self.assertNotIn("listen 443", self.conf)
        self.assertNotIn("ssl_certificate", self.conf)
        self.assertNotIn("return 301", self.conf)

    def test_api_key_and_rate_limit_kept(self) -> None:
        self.assertIn('set $api_key "Bearer key123";', self.conf)
        self.assertIn("if ($http_authorization != $api_key)", self.conf)
        self.assertIn("return 401", self.conf)
        self.assertIn("rate=10r/s", self.conf)
        self.assertIn("burst=20 nodelay", self.conf)
        self.assertIn("limit_req_status 429", self.conf)

    def test_rate_limit_per_device(self) -> None:
        # В LAN настоящие IP устройств видны напрямую: лимит по
        # $binary_remote_addr, map из туннеля не нужен
        self.assertIn(
            "limit_req_zone $binary_remote_addr zone=ollama_limit:10m",
            self.conf)
        self.assertNotIn("map $http_cf_connecting_ip", self.conf)

    def test_streaming_and_loopback_upstream(self) -> None:
        self.assertIn("server 127.0.0.1:11434;", self.conf)
        self.assertIn("proxy_buffering off;", self.conf)
        self.assertIn("proxy_read_timeout 300s;", self.conf)
        self.assertIn("chunked_transfer_encoding on;", self.conf)

    def test_custom_port(self) -> None:
        conf = build_nginx_conf("_", "k", "", "",
                                scheme="lan", listen_port=9443)
        self.assertIn("listen 0.0.0.0:9443;", conf)

    def test_no_key_mode(self) -> None:
        """LAN без API Key (доверенная сеть): блока auth нет."""
        conf = build_nginx_conf("_", "", "", "", scheme="lan",
                                listen_port=8080, use_api_key=False)
        self.assertNotIn("set $api_key", conf)
        self.assertNotIn("return 401", conf)
        self.assertNotIn("Authorization", conf)
        # Rate limit и остальная защита остаются
        self.assertIn("rate=10r/s", conf)
        self.assertIn("listen 0.0.0.0:8080;", conf)
        self.assertIn("БЕЗ API Key", conf)

    def test_web_root_serves_chat_page(self) -> None:
        conf = build_nginx_conf("_", "", "", "", scheme="lan",
                                listen_port=8080, use_api_key=False,
                                web_root="C:/out/remote_access")
        self.assertIn("location = / {", conf)
        self.assertIn("root C:/out/remote_access;", conf)
        self.assertIn("try_files /chat.html =404;", conf)
        # API по-прежнему проксируется
        self.assertIn("proxy_pass http://ollama_backend;", conf)

    def test_web_root_ignored_with_key_required(self) -> None:
        """Веб-чат и с ключом работает (браузер спросит ключ сам)."""
        conf = build_nginx_conf("_", "key123", "", "", scheme="lan",
                                listen_port=8080, use_api_key=True,
                                web_root="C:/out")
        self.assertIn('set $api_key "Bearer key123";', conf)
        self.assertIn("try_files /chat.html =404;", conf)

    def test_classic_ignores_use_api_key_false(self) -> None:
        """Вне LAN ключ обязателен даже при use_api_key=False."""
        conf = build_nginx_conf("x.org", "k", "c.pem", "c.key",
                                scheme="classic", use_api_key=False)
        self.assertIn('set $api_key "Bearer k";', conf)
        self.assertIn("return 401", conf)


class ChatPageTest(unittest.TestCase):
    """Страница веб-чата: содержание и запись файла."""

    def test_page_markers(self) -> None:
        from src.chat_page import CHAT_PAGE_HTML

        self.assertIn("<!DOCTYPE html", CHAT_PAGE_HTML)
        self.assertIn('charset="utf-8"', CHAT_PAGE_HTML)
        self.assertIn("viewport", CHAT_PAGE_HTML)
        # Нативный API Ollama за Nginx
        self.assertIn('"/api/tags"', CHAT_PAGE_HTML)
        self.assertIn('"/api/chat"', CHAT_PAGE_HTML)
        self.assertIn("stream: true", CHAT_PAGE_HTML)
        # Ключ (если сервер требует) хранится в localStorage
        self.assertIn("localStorage", CHAT_PAGE_HTML)
        self.assertIn("Bearer ", CHAT_PAGE_HTML)
        # Ответ модели — только textContent (XSS)
        self.assertIn("textContent", CHAT_PAGE_HTML)

    def test_write_chat_page(self) -> None:
        from src.chat_page import write_chat_page

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "chat.html"
            path = write_chat_page(out)
            self.assertEqual(path, out)
            content = out.read_text(encoding="utf-8")
        self.assertIn("loadModels", content)
        self.assertIn("</html>", content)


# ----------------------------------------------------------------------
# check_proxy: цепочка «приложение → Nginx → Ollama»
# ----------------------------------------------------------------------
class CheckProxyTest(unittest.TestCase):
    """Проверка соединения (httpx замокан)."""

    def test_ok_returns_model_count(self) -> None:
        response = _FakeHTTPResponse(200, {"models": [
            {"name": "qwen2.5:7b"}, {"name": "llama3.1:8b"},
            {"name": "nomic-embed-text"}]})
        with mock.patch.object(remote_access.httpx, "get",
                               return_value=response) as mocked:
            ok, message = check_proxy("https://myai.duckdns.org", "key")
        self.assertTrue(ok)
        self.assertIn("3", message)
        self.assertIn("/api/tags", mocked.call_args.args[0])
        self.assertEqual(mocked.call_args.kwargs["headers"],
                         {"Authorization": "Bearer key"})

    def test_unauthorized(self) -> None:
        response = _FakeHTTPResponse(401, text="Unauthorized")
        with mock.patch.object(remote_access.httpx, "get",
                               return_value=response):
            ok, message = check_proxy("https://x.org", "bad-key")
        self.assertFalse(ok)
        self.assertIn("401", message)

    def test_unexpected_status(self) -> None:
        response = _FakeHTTPResponse(502, text="Bad Gateway")
        with mock.patch.object(remote_access.httpx, "get",
                               return_value=response):
            ok, message = check_proxy("https://x.org", "key")
        self.assertFalse(ok)
        self.assertIn("502", message)

    def test_network_error(self) -> None:
        with mock.patch.object(remote_access.httpx, "get",
                               side_effect=OSError("refused")):
            ok, message = check_proxy("https://x.org", "key")
        self.assertFalse(ok)
        self.assertIn("недоступен", message)

    def test_empty_url(self) -> None:
        ok, message = check_proxy("", "key")
        self.assertFalse(ok)
        self.assertIn("не задан", message)

    def test_no_key_sends_no_auth_header(self) -> None:
        """Сервер LAN без API Key: заголовок не отправляется, 200 OK."""
        response = _FakeHTTPResponse(200, {"models": []})
        with mock.patch.object(remote_access.httpx, "get",
                               return_value=response) as mocked:
            ok, _ = check_proxy("http://192.168.1.50:8080", "")
        self.assertTrue(ok)
        self.assertIsNone(mocked.call_args.kwargs["headers"])


# ----------------------------------------------------------------------
# DDNS: протоколы провайдеров (httpx замокан)
# ----------------------------------------------------------------------
class DuckDNSProtocolTest(unittest.TestCase):
    """update_duckdns: URL, нормализация домена, разбор ответа."""

    def test_update_ok(self) -> None:
        response = _FakeHTTPResponse(200, text="OK")
        with mock.patch("src.ddns_client.httpx.get",
                        return_value=response) as mocked:
            self.assertTrue(update_duckdns("myai.duckdns.org", "tok"))
        self.assertIn("duckdns.org/update", mocked.call_args.args[0])
        params = mocked.call_args.kwargs["params"]
        self.assertEqual(params["domains"], "myai")
        self.assertEqual(params["token"], "tok")

    def test_update_rejected(self) -> None:
        response = _FakeHTTPResponse(200, text="KO")
        with mock.patch("src.ddns_client.httpx.get",
                        return_value=response):
            self.assertFalse(update_duckdns("myai", "bad-token"))

    def test_domain_normalization(self) -> None:
        self.assertEqual(duckdns_domain("myai.duckdns.org"), "myai")
        self.assertEqual(duckdns_domain("Myai.DuckDNS.org"), "myai")
        self.assertEqual(duckdns_domain("myai"), "myai")


class ApertoProtocolTest(unittest.TestCase):
    """update_aperto_dns: POST + Bearer + JSON."""

    def test_update_ok_and_rejected(self) -> None:
        ok_response = _FakeHTTPResponse(200)
        with mock.patch("src.ddns_client.httpx.post",
                        return_value=ok_response) as mocked:
            self.assertTrue(update_aperto_dns("x.apertodns.com", "key"))
        self.assertIn("apertodns.com/update", mocked.call_args.args[0])
        self.assertEqual(mocked.call_args.kwargs["headers"],
                         {"Authorization": "Bearer key"})
        self.assertEqual(mocked.call_args.kwargs["json"],
                         {"domain": "x.apertodns.com"})
        with mock.patch("src.ddns_client.httpx.post",
                        return_value=_FakeHTTPResponse(403)):
            self.assertFalse(update_aperto_dns("x.apertodns.com", "bad"))


class NoIPProtocolTest(unittest.TestCase):
    """update_noip: nic/update с Basic-аутентификацией."""

    def _update(self, text: str, status: int = 200) -> bool:
        response = _FakeHTTPResponse(status, text=text)
        with mock.patch("src.ddns_client.httpx.get",
                        return_value=response) as mocked:
            return update_noip("x.ddns.net", "user", "pass"), mocked

    def test_good_and_nochg_are_success(self) -> None:
        ok, mocked = self._update("good 93.184.216.34")
        self.assertTrue(ok)
        self.assertEqual(mocked.call_args.kwargs["auth"],
                         ("user", "pass"))
        self.assertEqual(mocked.call_args.kwargs["params"]["hostname"],
                         "x.ddns.net")
        ok, _ = self._update("nochg 93.184.216.34")
        self.assertTrue(ok)

    def test_failures(self) -> None:
        ok, _ = self._update("nohost")
        self.assertFalse(ok)
        ok, _ = self._update("badauth", status=401)
        self.assertFalse(ok)


class DDNSClientTest(unittest.TestCase):
    """Диспетчер провайдеров, ошибки без исключений, циклы."""

    def test_dispatch_providers(self) -> None:
        cases = (
            ("duckdns", "src.ddns_client.update_duckdns"),
            ("aperto", "src.ddns_client.update_aperto_dns"),
            ("noip", "src.ddns_client.update_noip"),
        )
        for provider, target in cases:
            client = DDNSClient(provider=provider, domain="x.org",
                                token="tok", secret="pass")
            with mock.patch(target, return_value=True) as mocked:
                ok, message = client.update_once()
            self.assertTrue(ok, message)
            self.assertIn("обновлён", message)
            mocked.assert_called_once()

    def test_unknown_provider(self) -> None:
        client = DDNSClient(provider="gopher", domain="x.org")
        ok, message = client.update_once()
        self.assertFalse(ok)
        self.assertIn("Неизвестный DDNS-провайдер", message)

    def test_network_error_returns_tuple(self) -> None:
        client = DDNSClient(provider="duckdns", domain="x.org", token="t")
        with mock.patch("src.ddns_client.update_duckdns",
                        side_effect=OSError("down")):
            ok, message = client.update_once()
        self.assertFalse(ok)
        self.assertIn("Ошибка DDNS", message)

    def test_noip_uses_secret_as_password(self) -> None:
        client = DDNSClient(provider="noip", domain="x.ddns.net",
                            token="user", secret="pass")
        with mock.patch("src.ddns_client.update_noip",
                        return_value=True) as mocked:
            ok, _ = client.update_once()
        self.assertTrue(ok)
        mocked.assert_called_once_with("x.ddns.net", "user", "pass")

    def test_run_forever_stops_on_max_iterations(self) -> None:
        client = DDNSClient(provider="duckdns", domain="x.org",
                            interval_minutes=0)
        with mock.patch.object(client, "update_once",
                               return_value=(True, "ok")) as mocked:
            client.run_forever(max_iterations=3)
        self.assertEqual(mocked.call_count, 3)

    def test_ddns_loop_counts_and_survives_errors(self) -> None:
        calls = []

        def _update() -> None:
            calls.append(1)
            if len(calls) == 1:
                raise OSError("первый раз — сеть моргнула")

        errors = []
        ddns_loop(_update, interval_minutes=0, max_iterations=3,
                  on_error=errors.append)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(errors), 1)


class DDNSCliTest(unittest.TestCase):
    """CLI: --run-once (для планировщика задач)."""

    def test_run_once_success(self) -> None:
        with mock.patch("src.ddns_client.DDNSClient") as cls:
            instance = cls.return_value
            instance.update_once.return_value = (True, "IP обновлён")
            code = DDNSCliTest._main([
                "--provider", "aperto", "--domain", "x.apertodns.com",
                "--token", "key", "--run-once"])
        self.assertEqual(code, 0)
        kwargs = cls.call_args.kwargs
        self.assertEqual(kwargs["provider"], "aperto")
        self.assertEqual(kwargs["domain"], "x.apertodns.com")
        self.assertEqual(kwargs["token"], "key")

    def test_run_once_failure_exit_code(self) -> None:
        with mock.patch.object(DDNSClient, "update_once",
                               return_value=(False, "отказ")):
            code = DDNSCliTest._main(
                ["--domain", "x.org", "--token", "t", "--run-once"])
        self.assertEqual(code, 1)

    @staticmethod
    def _main(argv):
        from src.ddns_client import main
        return main(argv)


# ----------------------------------------------------------------------
# OllamaProvider: прокси-режим
# ----------------------------------------------------------------------
class ProviderRemoteAccessTest(unittest.TestCase):
    """base_url/заголовки/таймауты провайдера при удалённом доступе."""

    def test_proxy_mode_endpoints_and_headers(self) -> None:
        provider = OllamaProvider(model="qwen2.5:7b",
                                  remote_access=dict(PROXY))
        self.assertTrue(provider.uses_proxy)
        self.assertEqual(provider.base_url, "https://myai.duckdns.org/v1")
        self.assertEqual(provider.api_base, "https://myai.duckdns.org")
        # Ключ Nginx уходит заголовком; ключ SDK — прежняя заглушка
        self.assertEqual(
            provider.client.default_headers.get("Authorization"),
            "Bearer " + "a" * 32)
        self.assertEqual(provider.client.api_key, "ollama")

    def test_proxy_url_normalization(self) -> None:
        for raw in ("https://x.org/", "https://x.org/v1/",
                    " https://x.org "):
            provider = OllamaProvider(model="qwen2.5:7b", remote_access={
                "proxy_url": raw, "api_key": "k"})
            self.assertEqual(provider.base_url, "https://x.org/v1")
            self.assertEqual(provider.api_base, "https://x.org")

    def test_local_mode_without_url_or_key(self) -> None:
        for settings in (None, {}, {"proxy_url": "https://x.org"},
                         {"api_key": "k"}):
            provider = OllamaProvider(model="qwen2.5:7b",
                                      remote_access=settings)
            self.assertFalse(provider.uses_proxy)
            self.assertEqual(provider.base_url,
                             "http://localhost:11434/v1")
            self.assertEqual(provider.api_base,
                             "http://localhost:11434")

    def test_proxy_without_key_when_server_has_no_auth(self) -> None:
        """LAN без API Key: достаточно URL (флаг use_api_key=False)."""
        provider = OllamaProvider(model="qwen2.5:7b", remote_access={
            "proxy_url": "http://192.168.1.50:8080", "use_api_key": False})
        self.assertTrue(provider.uses_proxy)
        self.assertEqual(provider.base_url, "http://192.168.1.50:8080/v1")
        self.assertEqual(provider.api_base, "http://192.168.1.50:8080")
        # Заголовка Authorization нет — сервер пускает без ключа
        self.assertNotIn("Authorization", provider.client.default_headers)
        # Словарь без флага (legacy/ручной) — прежнее правило «нужен ключ»
        legacy = OllamaProvider(model="qwen2.5:7b", remote_access={
            "proxy_url": "http://192.168.1.50:8080"})
        self.assertFalse(legacy.uses_proxy)

    def test_is_available_through_proxy(self) -> None:
        provider = OllamaProvider(model="qwen2.5:7b",
                                  remote_access=dict(PROXY))
        response = _FakeHTTPResponse(200, {"models": [
            {"name": "qwen2.5:7b"}]})
        with mock.patch("src.llm_providers.httpx.get",
                        return_value=response) as mocked:
            self.assertTrue(provider.is_available())
        self.assertIn("https://myai.duckdns.org/api/tags",
                      mocked.call_args.args[0])
        self.assertEqual(mocked.call_args.kwargs["headers"],
                         {"Authorization": "Bearer " + "a" * 32})
        # Прокси длиннее локальной петли: расширенный таймаут
        self.assertEqual(mocked.call_args.kwargs["timeout"],
                         PROXY_TAGS_TIMEOUT)

    def test_supports_tool_calling_sends_auth(self) -> None:
        provider = OllamaProvider(model="qwen2.5:7b",
                                  remote_access=dict(PROXY))
        response = _FakeHTTPResponse(200, {"capabilities": ["tools"]})
        with mock.patch("src.llm_providers.httpx.post",
                        return_value=response) as mocked:
            self.assertTrue(provider.supports_tool_calling())
        self.assertEqual(mocked.call_args.kwargs["headers"],
                         {"Authorization": "Bearer " + "a" * 32})

    def test_set_remote_access_hot_switch(self) -> None:
        provider = OllamaProvider(model="qwen2.5:7b")
        self.assertFalse(provider.uses_proxy)
        provider.set_remote_access(dict(PROXY))
        self.assertTrue(provider.uses_proxy)
        self.assertEqual(provider.base_url, "https://myai.duckdns.org/v1")
        self.assertEqual(provider.client.default_headers.get(
            "Authorization"), "Bearer " + "a" * 32)
        # Обратно на локальный режим: заголовок пропадает
        provider.set_remote_access({})
        self.assertFalse(provider.uses_proxy)
        self.assertEqual(provider.base_url,
                         "http://localhost:11434/v1")
        self.assertNotIn("Authorization",
                         provider.client.default_headers)

    def test_factory_passes_remote_access_to_ollama_only(self) -> None:
        provider = LLMProviderFactory.create(
            "ollama_qwen", remote_access=dict(PROXY))
        self.assertEqual(provider.base_url, "https://myai.duckdns.org/v1")
        cloud = LLMProviderFactory.create(
            "deepseek", api_key="k", remote_access=dict(PROXY))
        self.assertIsInstance(cloud, DeepSeekProvider)
        self.assertEqual(cloud.base_url, "https://api.deepseek.com")


# ----------------------------------------------------------------------
# Ядро: set_ollama_remote_access / switch_provider
# ----------------------------------------------------------------------
class AgentRemoteAccessTest(unittest.TestCase):
    """Хранение, горячее применение и перенос настроек ядром."""

    def setUp(self) -> None:
        self.core = DeepSeekAgent(api_key="test", mock_mode=True,
                                  storage=None)

    def test_set_sanitizes_and_stores(self) -> None:
        clean = self.core.set_ollama_remote_access({
            **PROXY, "мусор": 1, "rate_limit": "кривое"})
        self.assertEqual(clean["proxy_url"], "https://myai.duckdns.org")
        self.assertNotIn("мусор", clean)
        self.assertEqual(clean["rate_limit"], 10)
        self.assertEqual(self.core.ollama_remote_access, clean)

    def test_property_returns_copy(self) -> None:
        self.core.set_ollama_remote_access(dict(PROXY))
        snapshot = self.core.ollama_remote_access
        snapshot["proxy_url"] = "взломано"
        self.assertEqual(self.core.ollama_remote_access["proxy_url"],
                         "https://myai.duckdns.org")

    def test_without_provider_does_not_crash(self) -> None:
        self.assertIsNone(self.core.provider)
        clean = self.core.set_ollama_remote_access(dict(PROXY))
        self.assertEqual(clean["api_key"], "a" * 32)

    def test_hot_apply_to_active_provider(self) -> None:
        self.core.switch_provider("ollama_qwen")
        self.core.set_ollama_remote_access(dict(PROXY))
        provider = self.core.provider
        self.assertTrue(provider.uses_proxy)
        self.assertEqual(provider.base_url, "https://myai.duckdns.org/v1")

    def test_switch_provider_carries_settings(self) -> None:
        self.core.set_ollama_remote_access(dict(PROXY))
        self.core.switch_provider("ollama_llama")
        self.assertEqual(self.core.provider.base_url,
                         "https://myai.duckdns.org/v1")

    def test_deepseek_ignores_remote_access(self) -> None:
        self.core.set_ollama_remote_access(dict(PROXY))
        self.core.switch_provider("deepseek", api_key="k")
        self.assertIsInstance(self.core.provider, DeepSeekProvider)
        self.assertEqual(self.core.provider.base_url,
                         "https://api.deepseek.com")
        # Настройки сохранены — при возврате на Ollama применятся
        self.core.switch_provider("ollama_qwen")
        self.assertEqual(self.core.provider.base_url,
                         "https://myai.duckdns.org/v1")

    def test_clearing_settings_returns_to_local(self) -> None:
        self.core.switch_provider("ollama_qwen")
        self.core.set_ollama_remote_access(dict(PROXY))
        self.assertTrue(self.core.provider.uses_proxy)
        self.core.set_ollama_remote_access(None)
        self.assertFalse(self.core.provider.uses_proxy)
        self.assertEqual(self.core.provider.base_url,
                         "http://localhost:11434/v1")


# ----------------------------------------------------------------------
# UI-панель (офлайн, page=None)
# ----------------------------------------------------------------------
class RemoteAccessPanelTest(unittest.TestCase):
    """Панель «Удалённый доступ»: применение, генерации, статусы."""

    def setUp(self) -> None:
        from app.ui.remote_access_view import RemoteAccessPanel

        self.panel_cls = RemoteAccessPanel
        self.app_state = _StubAppState()
        self.agent = make_agent()
        self.app = make_app(self.agent)
        self.app.app_state = self.app_state

    def _panel(self):
        return self.panel_cls(self.app)

    def test_panel_builds_offline(self) -> None:
        panel = self._panel()
        container = panel.build()
        self.assertIsNotNone(container)
        self.assertEqual(panel.ddns_dropdown.value, "duckdns")

    def test_fields_prefilled_from_store(self) -> None:
        self.app_state.saved[REMOTE_ACCESS_KEY] = json.dumps(
            RemoteAccessSettings.sanitize(
                {**PROXY, "ddns_domain": "myai.duckdns.org"}))
        panel = self._panel()
        self.assertEqual(panel.proxy_url_field.value,
                         "https://myai.duckdns.org")
        self.assertEqual(panel.api_key_field.value, "a" * 32)
        self.assertEqual(panel.ddns_domain_field.value,
                         "myai.duckdns.org")

    def test_blur_applies_to_core_and_store(self) -> None:
        panel = self._panel()
        panel.proxy_url_field.value = "https://myai.duckdns.org/"
        panel.api_key_field.value = "k" * 32
        panel._on_field_blur(None)
        saved = json.loads(self.app_state.saved[REMOTE_ACCESS_KEY])
        self.assertEqual(saved["proxy_url"], "https://myai.duckdns.org")
        core = self.agent.core
        self.assertEqual(core.ollama_remote_access["api_key"], "k" * 32)
        self.assertIn("применены", panel.status_text.value)

    def test_generate_key_button(self) -> None:
        panel = self._panel()
        panel._on_generate_key(None)
        self.assertRegex(panel.api_key_field.value, r"^[0-9a-f]{32}$")
        saved = json.loads(self.app_state.saved[REMOTE_ACCESS_KEY])
        self.assertEqual(saved["api_key"], panel.api_key_field.value)

    def test_ddns_provider_change_saved(self) -> None:
        panel = self._panel()
        event = SimpleNamespace(
            control=SimpleNamespace(value="aperto"))
        panel._on_ddns_change(event)
        saved = json.loads(self.app_state.saved[REMOTE_ACCESS_KEY])
        self.assertEqual(saved["ddns_provider"], "aperto")

    def test_check_connection_status(self) -> None:
        panel = self._panel()
        panel.proxy_url_field.value = "https://myai.duckdns.org"
        panel.api_key_field.value = "key"
        response = _FakeHTTPResponse(200, {"models": [
            {"name": "m1"}, {"name": "m2"}]})
        with mock.patch.object(remote_access.httpx, "get",
                               return_value=response):
            panel._check_connection()
        self.assertIn("2", panel.status_text.value)
        self.assertNotEqual(panel.status_text.color, ft_color_red())

    def test_check_connection_error_status(self) -> None:
        panel = self._panel()
        panel.proxy_url_field.value = "https://myai.duckdns.org"
        panel.api_key_field.value = "bad"
        response = _FakeHTTPResponse(401)
        with mock.patch.object(remote_access.httpx, "get",
                               return_value=response):
            panel._check_connection()
        self.assertIn("401", panel.status_text.value)
        self.assertEqual(panel.status_text.color, ft_color_red())

    def test_update_ddns_requires_domain_and_token(self) -> None:
        panel = self._panel()
        panel._update_ddns_now()
        self.assertIn("Заполните", panel.status_text.value)

    def test_update_ddns_calls_client(self) -> None:
        panel = self._panel()
        panel.ddns_dropdown.value = "duckdns"
        panel.ddns_domain_field.value = "myai.duckdns.org"
        panel.ddns_token_field.value = "tok"
        client = mock.Mock()
        client.update_once.return_value = (True, "IP обновлён")
        with mock.patch("app.ui.remote_access_view.DDNSClient",
                        return_value=client) as mocked:
            panel._update_ddns_now()
        self.assertIn("IP обновлён", panel.status_text.value)
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs["provider"], "duckdns")
        self.assertEqual(kwargs["domain"], "myai.duckdns.org")
        self.assertEqual(kwargs["token"], "tok")

    def test_generate_nginx_writes_file(self) -> None:
        panel = self._panel()
        panel.proxy_url_field.value = "https://myai.duckdns.org"
        panel.api_key_field.value = "k" * 32
        panel.ddns_domain_field.value = "myai.duckdns.org"
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("app.ui.remote_access_view.default_output_dir",
                           return_value=Path(tmp)):
            panel._generate_nginx()
            out = Path(tmp) / "nginx.conf"
            self.assertTrue(out.exists())
            content = out.read_text(encoding="utf-8")
        self.assertIn("server_name myai.duckdns.org;", content)
        self.assertIn("Bearer " + "k" * 32, content)
        # Пути Certbot подставлены по домену
        self.assertEqual(panel.cert_path_field.value,
                         "C:/Certbot/live/myai.duckdns.org/fullchain.pem")
        self.assertIn("nginx.conf записан", panel.status_text.value)

    def test_generate_nginx_requires_domain(self) -> None:
        panel = self._panel()
        panel._generate_nginx()
        self.assertIn("DDNS-домен", panel.status_text.value)
        self.assertEqual(panel.status_text.color, ft_color_red())

    def test_generate_nginx_requires_key(self) -> None:
        panel = self._panel()
        panel.ddns_domain_field.value = "myai.duckdns.org"
        panel._generate_nginx()
        self.assertIn("API Key", panel.status_text.value)

    def test_mode_indicator(self) -> None:
        panel = self._panel()
        self.assertIn("локально", panel.mode_text.value)
        panel.proxy_url_field.value = "https://x.org"
        panel.api_key_field.value = "k"
        panel._on_field_blur(None)
        self.assertIn("через прокси", panel.mode_text.value)

    def test_mode_indicator_shows_tunnel(self) -> None:
        panel = self._panel()
        panel.scheme_dropdown.value = "tunnel"
        panel.proxy_url_field.value = "https://ai.example.com"
        panel.api_key_field.value = "k"
        panel._on_field_blur(None)
        self.assertIn("через прокси", panel.mode_text.value)
        self.assertIn("Cloudflare Tunnel", panel.mode_text.value)

    def test_mode_indicator_shows_lan(self) -> None:
        panel = self._panel()
        panel.scheme_dropdown.value = "lan"
        panel.proxy_url_field.value = "http://192.168.1.50:8080"
        panel.api_key_field.value = "k"
        panel._on_field_blur(None)
        self.assertIn("через прокси", panel.mode_text.value)
        self.assertIn("локальная сеть", panel.mode_text.value)

    # ------------------------------------------------------------------
    # Схема доступа: classic / tunnel / lan
    # ------------------------------------------------------------------
    def _scheme_event(self, value: str):
        return SimpleNamespace(control=SimpleNamespace(value=value))

    def test_scheme_change_saves_and_locks_fields(self) -> None:
        panel = self._panel()
        # Tunnel и LAN: DDNS/сертификаты блокируются, порт доступен
        for scheme in ("tunnel", "lan"):
            panel._on_scheme_change(self._scheme_event(scheme))
            saved = json.loads(self.app_state.saved[REMOTE_ACCESS_KEY])
            self.assertEqual(saved["scheme"], scheme)
            self.assertIn("Схема", panel.status_text.value)
            self.assertTrue(panel.ddns_dropdown.disabled)
            self.assertTrue(panel.cert_path_field.disabled)
            self.assertFalse(panel.listen_port_field.disabled)
        # Классика: наоборот
        panel._on_scheme_change(self._scheme_event("classic"))
        self.assertFalse(panel.ddns_dropdown.disabled)
        self.assertFalse(panel.cert_path_field.disabled)
        self.assertTrue(panel.listen_port_field.disabled)

    def test_scheme_settings_restored_from_store(self) -> None:
        # В том числе легаси-формат ai_1_30-tunnel
        self.app_state.saved[REMOTE_ACCESS_KEY] = json.dumps(
            RemoteAccessSettings.sanitize(
                {"tunnel_mode": True, "tunnel_port": 9443}))
        panel = self._panel()
        self.assertEqual(panel.scheme_dropdown.value, "tunnel")
        self.assertEqual(panel.listen_port_field.value, "9443")
        self.assertTrue(panel.ddns_dropdown.disabled)
        self.assertFalse(panel.listen_port_field.disabled)

    def test_generate_nginx_tunnel_mode(self) -> None:
        panel = self._panel()
        panel.scheme_dropdown.value = "tunnel"
        panel.listen_port_field.value = "8080"
        panel.proxy_url_field.value = "https://ai.example.com"
        panel.api_key_field.value = "k" * 32
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("app.ui.remote_access_view.default_output_dir",
                           return_value=Path(tmp)):
            panel._generate_nginx()
            out = Path(tmp) / "nginx.conf"
            self.assertTrue(out.exists())
            content = out.read_text(encoding="utf-8")
        # HTTP на 127.0.0.1, без сертификатов; hostname — из URL прокси
        self.assertIn("listen 127.0.0.1:8080;", content)
        self.assertIn("server_name ai.example.com;", content)
        self.assertNotIn("ssl_certificate", content)
        self.assertIn("Bearer " + "k" * 32, content)
        self.assertEqual(panel.cert_path_field.value, "")
        self.assertIn("cloudflared", panel.status_text.value)
        saved = json.loads(self.app_state.saved[REMOTE_ACCESS_KEY])
        self.assertEqual(saved["scheme"], "tunnel")
        self.assertEqual(saved["listen_port"], 8080)

    def test_generate_nginx_lan_mode(self) -> None:
        panel = self._panel()
        panel.scheme_dropdown.value = "lan"
        panel.listen_port_field.value = "8080"
        panel.api_key_field.value = "k" * 32
        # Домен и URL прокси не нужны вовсе
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("app.ui.remote_access_view.default_output_dir",
                           return_value=Path(tmp)):
            panel._generate_nginx()
            out = Path(tmp) / "nginx.conf"
            self.assertTrue(out.exists())
            content = out.read_text(encoding="utf-8")
        self.assertIn("listen 0.0.0.0:8080;", content)
        self.assertIn("server_name _;", content)
        self.assertNotIn("ssl_certificate", content)
        self.assertIn("Bearer " + "k" * 32, content)
        # Сертификаты не подставляются; статус подсказывает про LAN
        self.assertEqual(panel.cert_path_field.value, "")
        self.assertIn("брандмауэре", panel.status_text.value)
        saved = json.loads(self.app_state.saved[REMOTE_ACCESS_KEY])
        self.assertEqual(saved["scheme"], "lan")
        self.assertEqual(saved["listen_port"], 8080)

    # ------------------------------------------------------------------
    # LAN без API Key + веб-чат
    # ------------------------------------------------------------------
    def test_use_key_switch_sync_per_scheme(self) -> None:
        panel = self._panel()
        # Вне LAN свитч заблокирован и включён (ключ обязателен)
        self.assertTrue(panel.use_key_switch.disabled)
        self.assertTrue(panel.use_key_switch.value)
        panel._on_scheme_change(self._scheme_event("lan"))
        self.assertFalse(panel.use_key_switch.disabled)
        # Выключили ключ: поля ключа блокируются, настройка сохраняется
        panel.use_key_switch.value = False
        panel._on_use_key_change(None)
        saved = json.loads(self.app_state.saved[REMOTE_ACCESS_KEY])
        self.assertFalse(saved["use_api_key"])
        self.assertTrue(panel.api_key_field.disabled)
        self.assertTrue(panel.generate_key_button.disabled)
        # Обратно в классику: свитч принудительно включается
        panel._on_scheme_change(self._scheme_event("classic"))
        self.assertTrue(panel.use_key_switch.value)
        self.assertFalse(panel.api_key_field.disabled)

    def test_generate_nginx_lan_no_key_with_chat_page(self) -> None:
        """LAN без ключа: конфиг без auth + страница чата по /."""
        panel = self._panel()
        panel.scheme_dropdown.value = "lan"
        panel.listen_port_field.value = "8080"
        panel.use_key_switch.value = False
        # Ни ключа, ни URL прокси не нужно — генерация проходит
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("app.ui.remote_access_view.default_output_dir",
                           return_value=Path(tmp)):
            panel._generate_nginx()
            conf = (Path(tmp) / "nginx.conf").read_text(encoding="utf-8")
            page = (Path(tmp) / "chat.html").read_text(encoding="utf-8")
        self.assertIn("listen 0.0.0.0:8080;", conf)
        self.assertNotIn("set $api_key", conf)
        self.assertIn("try_files /chat.html =404;", conf)
        self.assertIn("/api/chat", page)
        self.assertIn("chat.html", panel.status_text.value)
        self.assertIn("браузере", panel.status_text.value)
        saved = json.loads(self.app_state.saved[REMOTE_ACCESS_KEY])
        self.assertFalse(saved["use_api_key"])

    def test_on_show_reloads_settings(self) -> None:
        panel = self._panel()
        self.assertEqual(panel.proxy_url_field.value, "")
        self.app_state.saved[REMOTE_ACCESS_KEY] = json.dumps(
            RemoteAccessSettings.sanitize(dict(PROXY)))
        panel.on_show()
        self.assertEqual(panel.proxy_url_field.value,
                         "https://myai.duckdns.org")


def ft_color_red():
    """Красный статус ошибки в панели (как в остальных панелях)."""
    import flet as ft
    return ft.Colors.RED_600


if __name__ == "__main__":
    unittest.main()
