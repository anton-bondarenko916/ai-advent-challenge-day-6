"""Панель «Удалённый доступ (Ollama как сетевой сервис)» (ТЗ ai_1_30).

Вкладывается во вкладку «Настройки» под панелью оптимизации. Логика:

* «URL прокси» + «API Key»: заданы оба — OllamaProvider ходит через
  Nginx-прокси с заголовком ``Authorization: Bearer``; пустые поля —
  локальный режим без изменений;
* «Схема доступа» (дропдаун): классическая (DDNS + проброс 443/80 +
  HTTPS с Certbot), Cloudflare Tunnel (Nginx в HTTP-режиме на
  127.0.0.1, TLS — на edge Cloudflare, cloudflared соединяет edge с
  локальным портом — DDNS/сертификаты/проброс не нужны) и локальная
  сеть (Nginx на 0.0.0.0 по HTTP — без домена, доступ только из своей
  сети); по схеме блокируются ненужные поля;
* «Сгенерировать» — ключ для Nginx (secrets.token_hex(16), не вручную);
* «Проверить соединение» — GET {прокси}/api/tags с ключом (список
  моделей = цепочка «приложение → Nginx → Ollama» жива);
* DDNS-блок (классическая схема): провайдер (DuckDNS/ApertoDNS/No-IP),
  домен, токен и кнопка «Обновить IP сейчас» (постоянно IP обновляет
  внешний DDNS-клиент — src/ddns_client.py как задача планировщика
  Windows);
* блок Nginx: rate limit/burst и порт → кнопка
  «Сгенерировать nginx.conf» пишет готовый конфиг в
  output/remote_access/nginx.conf;
* сохранение глобальное — таблица app_state (ключ ollama_remote_access),
  переживает перезапуск; применение — set_ollama_remote_access ядра.

Офлайн-режим (page=None в тестах): изменения применяются к ядру и
хранилищу сразу, обновления UI и фоновые задачи не выполняются.
Применение настроек — на blur полей (как в панели оптимизации).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import flet as ft

from app.ui import components
from src.chat_page import write_chat_page
from src.ddns_client import DDNSClient
from src.remote_access import (
    SCHEME_CLASSIC,
    SCHEME_LABELS,
    SCHEME_LAN,
    SCHEME_TUNNEL,
    RemoteAccessSettings,
    check_proxy,
    default_cert_paths,
    default_output_dir,
    generate_api_key,
    normalize_host,
    proxy_active,
    write_nginx_conf,
)


class RemoteAccessPanel:
    """Панель удалённого доступа (секция вкладки «Настройки»)."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.agent = app.agent
        self.settings: Dict[str, Any] = self._load_settings()

        # Схема доступа: классическая (DDNS + проброс + HTTPS),
        # Cloudflare Tunnel (Nginx на 127.0.0.1 за cloudflared) или
        # локальная сеть (Nginx на 0.0.0.0, HTTP, без домена)
        self.scheme_dropdown = ft.Dropdown(
            label="Схема доступа",
            width=360,
            options=[
                ft.DropdownOption(key=key, content=ft.Text(label))
                for key, label in SCHEME_LABELS.items()
            ],
            value=str(self.settings.get("scheme") or SCHEME_CLASSIC),
            on_select=self._on_scheme_change,
        )
        self.listen_port_field = ft.TextField(
            label="Порт Nginx",
            value=str(self.settings.get("listen_port", "")),
            width=150,
            on_blur=self._on_field_blur,
        )
        # Отключение ключа имеет смысл только в LAN (доверенная сеть):
        # тогда браузерный чат по / открывается без API Key
        self.use_key_switch = ft.Switch(
            label="Требовать API Key",
            value=bool(self.settings.get("use_api_key", True)),
            on_change=self._on_use_key_change,
            tooltip="Отключать можно только для схемы LAN (доверенная "
                    "сеть): браузерный чат откроется без ключа",
        )

        self.proxy_url_field = ft.TextField(
            label="URL прокси",
            hint_text="Например, https://myai.duckdns.org",
            value=str(self.settings.get("proxy_url") or ""),
            width=340,
            on_blur=self._on_field_blur,
        )
        self.api_key_field = ft.TextField(
            label="API Key",
            value=str(self.settings.get("api_key") or ""),
            password=True,
            can_reveal_password=True,
            width=280,
            on_blur=self._on_field_blur,
        )
        self.generate_key_button = ft.FilledTonalButton(
            "🔑 Сгенерировать",
            on_click=self._on_generate_key,
            tooltip="Новый ключ для Nginx (рандом, не вручную)",
        )
        self.check_button = ft.OutlinedButton(
            "🔌 Проверить соединение",
            on_click=self._on_check,
            tooltip="GET {прокси}/api/tags с ключом: должны вернуться модели",
        )
        self.mode_text = ft.Text("", size=12, expand=True)

        self.ddns_dropdown = ft.Dropdown(
            label="DDNS-провайдер",
            width=180,
            options=[
                ft.DropdownOption(key=key, content=ft.Text(name))
                for key, name in RemoteAccessSettings.DDNS_PROVIDERS
            ],
            value=str(self.settings.get("ddns_provider") or "duckdns"),
            on_select=self._on_ddns_change,
        )
        self.ddns_domain_field = ft.TextField(
            label="DDNS-домен",
            hint_text="myai.duckdns.org",
            value=str(self.settings.get("ddns_domain") or ""),
            width=240,
            on_blur=self._on_field_blur,
        )
        self.ddns_token_field = ft.TextField(
            label="Token / ключ DDNS",
            value=str(self.settings.get("ddns_token") or ""),
            password=True,
            can_reveal_password=True,
            width=240,
            on_blur=self._on_field_blur,
        )
        self.ddns_update_button = ft.FilledTonalButton(
            "🔄 Обновить IP сейчас",
            on_click=self._on_update_ddns,
            tooltip="Постоянно IP обновляет внешний DDNS-клиент "
                    "(планировщик задач) — см. docs/remote_access.md",
        )

        self.rate_limit_field = ft.TextField(
            label="Rate limit (r/s)",
            value=str(self.settings.get("rate_limit", "")),
            width=150,
            on_blur=self._on_field_blur,
        )
        self.burst_field = ft.TextField(
            label="Burst",
            value=str(self.settings.get("burst", "")),
            width=120,
            on_blur=self._on_field_blur,
        )
        self.cert_path_field = ft.TextField(
            label="Сертификат (fullchain.pem)",
            value=str(self.settings.get("cert_path") or ""),
            width=430,
            on_blur=self._on_field_blur,
        )
        self.cert_key_path_field = ft.TextField(
            label="Ключ сертификата (privkey.pem)",
            value=str(self.settings.get("cert_key_path") or ""),
            width=430,
            on_blur=self._on_field_blur,
        )
        self.nginx_button = ft.FilledButton(
            "🌐 Сгенерировать nginx.conf",
            icon=ft.Icons.SECURITY,
            on_click=self._on_nginx,
        )
        self.status_text = ft.Text("", size=12, expand=True, selectable=True)

        self._sync_scheme_state()
        self._sync_key_fields()
        self._sync_mode_text()

    # ------------------------------------------------------------------
    # Компоновка
    # ------------------------------------------------------------------
    def build(self) -> ft.Container:
        """Собирает панель (секция с рамкой во вкладке «Настройки»)."""
        return ft.Container(
            content=ft.Column([
                ft.Text("Удалённый доступ (Ollama как сетевой сервис)",
                        size=15, weight=ft.FontWeight.BOLD),
                ft.Text(
                    "Ollama остаётся на 127.0.0.1 — наружу смотрит только "
                    "Nginx (API Key + rate limit). Схемы: классическая "
                    "(DDNS + проброс 443/80), Cloudflare Tunnel (без DDNS "
                    "и проброса) или локальная сеть (без домена). "
                    "Подробности: docs/remote_access.md",
                    size=12, color=ft.Colors.BLUE_GREY_500),
                self.scheme_dropdown,
                ft.Row([
                    self.proxy_url_field,
                    self.api_key_field,
                    self.generate_key_button,
                ], spacing=12, wrap=True),
                ft.Row([self.check_button, self.mode_text], spacing=12),
                ft.Text("DDNS: домен, который всегда указывает на ваш IP "
                        "(в схемах Tunnel и LAN не нужен)",
                        size=13, weight=ft.FontWeight.BOLD),
                ft.Row([
                    self.ddns_dropdown,
                    self.ddns_domain_field,
                    self.ddns_token_field,
                    self.ddns_update_button,
                ], spacing=12, wrap=True),
                ft.Text("Nginx", size=13, weight=ft.FontWeight.BOLD),
                ft.Row([
                    self.rate_limit_field,
                    self.burst_field,
                    self.listen_port_field,
                    self.use_key_switch,
                    self.nginx_button,
                ], spacing=12, wrap=True),
                self.cert_path_field,
                self.cert_key_path_field,
                ft.Row([self.status_text], spacing=12),
            ], spacing=10),
            border=ft.Border.all(1, ft.Colors.BLUE_GREY_200),
            border_radius=8,
            padding=14,
        )

    def on_show(self) -> None:
        """Перечитывает настройки при открытии вкладки (ядро — источник)."""
        self.settings = self._load_settings()
        self._sync_fields_from_settings()
        self.update()

    # ------------------------------------------------------------------
    # Загрузка/применение настроек
    # ------------------------------------------------------------------
    def _core_agent(self):
        """Ядро DeepSeekAgent (в legacy-заглушках тестов — сам agent)."""
        return getattr(self.agent, "core", self.agent)

    def _store(self):
        """Хранилище app_state (None — сохранение отключено)."""
        return getattr(self.app, "app_state", None)

    def _load_settings(self) -> Dict[str, Any]:
        return RemoteAccessSettings.load(self._store())

    def _collect_settings(self) -> Dict[str, Any]:
        """Собирает настройки из полей UI (числа — с откатом к текущим)."""
        def _int_field(field: Any, key: str) -> int:
            try:
                return int(float(str(field.value or "").replace(",", ".")))
            except (TypeError, ValueError):
                return int(self.settings.get(key,
                           RemoteAccessSettings.DEFAULTS[key]))

        return {
            "proxy_url": str(self.proxy_url_field.value or ""),
            "api_key": str(self.api_key_field.value or ""),
            "scheme": str(self.scheme_dropdown.value or SCHEME_CLASSIC),
            "listen_port": _int_field(self.listen_port_field,
                                      "listen_port"),
            "use_api_key": bool(self.use_key_switch.value),
            "ddns_provider": str(self.ddns_dropdown.value or "duckdns"),
            "ddns_domain": str(self.ddns_domain_field.value or ""),
            "ddns_token": str(self.ddns_token_field.value or ""),
            "ddns_secret": str(self.settings.get("ddns_secret") or ""),
            "ddns_interval_minutes": int(
                self.settings.get("ddns_interval_minutes",
                                  RemoteAccessSettings.DEFAULTS[
                                      "ddns_interval_minutes"])),
            "rate_limit": _int_field(self.rate_limit_field, "rate_limit"),
            "burst": _int_field(self.burst_field, "burst"),
            "cert_path": str(self.cert_path_field.value or ""),
            "cert_key_path": str(self.cert_key_path_field.value or ""),
            "ollama_host": str(self.settings.get("ollama_host") or ""),
        }

    def _apply(self, settings: Dict[str, Any], message: str = "") -> None:
        """Санитизирует, сохраняет в app_state и применяет к ядру."""
        self.settings = RemoteAccessSettings.sanitize(settings)
        RemoteAccessSettings.save(self._store(), self.settings)
        try:
            self._core_agent().set_ollama_remote_access(self.settings)
        except Exception as exc:  # noqa: BLE001 — legacy-заглушки тестов
            self._set_status(f"Не применено: {exc}", error=True)
            self.update()
            return
        self._sync_mode_text()
        if message:
            self._set_status(message)
        self.update()

    def _on_field_blur(self, event=None) -> None:
        self._apply(self._collect_settings(), "Настройки удалённого "
                    "доступа применены")

    def _sync_fields_from_settings(self) -> None:
        """Выставляет значения полей из self.settings (загрузка/сброс)."""
        self.proxy_url_field.value = str(
            self.settings.get("proxy_url") or "")
        self.api_key_field.value = str(self.settings.get("api_key") or "")
        self.scheme_dropdown.value = str(
            self.settings.get("scheme") or SCHEME_CLASSIC)
        self.listen_port_field.value = str(
            self.settings.get("listen_port", ""))
        self.use_key_switch.value = bool(
            self.settings.get("use_api_key", True))
        self.ddns_dropdown.value = str(
            self.settings.get("ddns_provider") or "duckdns")
        self.ddns_domain_field.value = str(
            self.settings.get("ddns_domain") or "")
        self.ddns_token_field.value = str(
            self.settings.get("ddns_token") or "")
        self.rate_limit_field.value = str(
            self.settings.get("rate_limit", ""))
        self.burst_field.value = str(self.settings.get("burst", ""))
        self.cert_path_field.value = str(
            self.settings.get("cert_path") or "")
        self.cert_key_path_field.value = str(
            self.settings.get("cert_key_path") or "")
        self._sync_scheme_state()
        self._sync_key_fields()
        self._sync_mode_text()

    def _sync_scheme_state(self) -> None:
        """Поля по схеме: в классической нужны DDNS и сертификаты
        (порт Nginx — стандартные 80/443), в Tunnel и LAN — наоборот:
        DDNS/сертификаты не нужны, но нужен порт. Свитч ключа
        отключаем только в LAN; вне LAN ключ обязателен всегда."""
        scheme = str(self.scheme_dropdown.value or SCHEME_CLASSIC)
        classic = scheme == SCHEME_CLASSIC
        lan = scheme == SCHEME_LAN
        self.listen_port_field.disabled = classic
        self.use_key_switch.disabled = not lan
        if not lan:
            self.use_key_switch.value = True
        for field in (self.ddns_dropdown, self.ddns_domain_field,
                      self.ddns_token_field, self.ddns_update_button,
                      self.cert_path_field, self.cert_key_path_field):
            field.disabled = not classic

    def _sync_key_fields(self) -> None:
        """Без требования ключа поля ключа не нужны (блокируются)."""
        need_key = bool(self.use_key_switch.value)
        self.api_key_field.disabled = not need_key
        self.generate_key_button.disabled = not need_key

    def _sync_mode_text(self) -> None:
        """Индикатор режима: через прокси (заданы URL и ключ) / локально."""
        if proxy_active(self.settings):
            suffix = {
                SCHEME_TUNNEL: " · Cloudflare Tunnel",
                SCHEME_LAN: " · локальная сеть",
            }.get(str(self.settings.get("scheme") or ""), "")
            self.mode_text.value = (
                f"● Режим: через прокси {self.settings['proxy_url']}"
                f"{suffix}")
            self.mode_text.color = ft.Colors.GREEN_700
        else:
            self.mode_text.value = (
                "● Режим: локально (заполните URL прокси И API Key, "
                "чтобы ходить через Nginx)")
            self.mode_text.color = ft.Colors.BLUE_GREY_500

    # ------------------------------------------------------------------
    # Обработчики
    # ------------------------------------------------------------------
    def _on_scheme_change(self, event=None) -> None:
        value = getattr(event.control, "value", None) if event is not None \
            else self.scheme_dropdown.value
        if not value:
            return
        # В живом Flet значение уже выставлено фреймворком; в тестах
        # (симуляция события) синхронизируем вручную
        self.scheme_dropdown.value = str(value)
        self._sync_scheme_state()
        self._sync_key_fields()
        label = SCHEME_LABELS.get(str(value), str(value))
        self._apply(self._collect_settings(), f"Схема: {label}")

    def _on_use_key_change(self, event=None) -> None:
        self._sync_key_fields()
        mode = ("с API Key" if self.use_key_switch.value
                else "БЕЗ API Key (доверенная сеть)")
        self._apply(self._collect_settings(), f"Nginx {mode}")
    def _on_generate_key(self, event=None) -> None:
        self.api_key_field.value = generate_api_key()
        self._apply(self._collect_settings(),
                    "Новый API Key сгенерирован — перегенерируйте nginx.conf")

    def _on_ddns_change(self, event=None) -> None:
        value = getattr(event.control, "value", None) if event is not None \
            else self.ddns_dropdown.value
        if not value:
            return
        # В живом Flet значение уже выставлено фреймворком; в тестах
        # (симуляция события) синхронизируем вручную
        self.ddns_dropdown.value = str(value)
        self._apply(self._collect_settings(),
                    f"DDNS-провайдер: {value}")

    def _on_check(self, event=None) -> None:
        self._schedule(self._check_async)

    def _on_update_ddns(self, event=None) -> None:
        self._schedule(self._update_ddns_async)

    def _on_nginx(self, event=None) -> None:
        self._schedule(self._nginx_async)

    # ------------------------------------------------------------------
    # Рабочие методы (синхронные — вызываются из async-обёрток в потоке
    # и напрямую из офлайн-тестов)
    # ------------------------------------------------------------------
    def _check_connection(self) -> None:
        """Проверка цепочки «приложение → Nginx → Ollama»."""
        self._set_status("Проверка соединения…")
        self.update()
        ok, message = check_proxy(
            str(self.proxy_url_field.value or ""),
            str(self.api_key_field.value or ""))
        self._set_status(message, error=not ok)
        self.update()

    def _update_ddns_now(self) -> None:
        """Разовое обновление IP у выбранного DDNS-провайдера."""
        provider = str(self.ddns_dropdown.value or "duckdns")
        domain = str(self.ddns_domain_field.value or "").strip()
        token = str(self.ddns_token_field.value or "").strip()
        if not domain or not token:
            self._set_status(
                "Заполните DDNS-домен и токен (токен/пароль)", error=True)
            self.update()
            return
        self._set_status(f"Обновление IP у {provider}…")
        self.update()
        ok, message = DDNSClient(
            provider=provider, domain=domain, token=token,
            secret=str(self.settings.get("ddns_secret") or ""),
        ).update_once()
        self._set_status(message, error=not ok)
        self.update()

    def _generate_nginx(self) -> None:
        """Пишет nginx.conf (и в LAN — страницу чата) в
        output/remote_access/ по выбранной схеме: классика — Certbot-пути
        и обязательный ключ; Tunnel — HTTP на 127.0.0.1; LAN — HTTP на
        0.0.0.0, ключ опционален, по / отдаётся веб-чат."""
        scheme = str(self.scheme_dropdown.value or SCHEME_CLASSIC)
        proxy_host = normalize_host(str(self.proxy_url_field.value or ""))
        ddns_host = normalize_host(str(self.ddns_domain_field.value or ""))
        api_key = str(self.api_key_field.value or "").strip()
        use_key = bool(self.use_key_switch.value) or scheme != SCHEME_LAN
        # Классика: hostname — DDNS-домен (или из URL); Tunnel: hostname
        # Cloudflare из URL прокси; LAN: домен не нужен (server_name _)
        if scheme == SCHEME_LAN:
            domain = "_"
        elif scheme == SCHEME_TUNNEL:
            domain = proxy_host or ddns_host
        else:
            domain = ddns_host or proxy_host
        if not domain:
            self._set_status(
                "Укажите URL прокси (или DDNS-домен) — hostname нужен "
                "для server_name", error=True)
            self.update()
            return
        if use_key and not api_key:
            self._set_status(
                "Сгенерируйте API Key (или выключите «Требовать API Key» "
                "в схеме LAN для доверенной сети)", error=True)
            self.update()
            return
        if (scheme == SCHEME_CLASSIC
                and not str(self.cert_path_field.value or "").strip()):
            cert_path, cert_key_path = default_cert_paths(domain)
            self.cert_path_field.value = cert_path
            self.cert_key_path_field.value = cert_key_path
        self._apply(self._collect_settings())
        out_dir = default_output_dir()
        path = write_nginx_conf(
            out_dir / "nginx.conf",
            domain=domain,
            api_key=self.settings["api_key"],
            cert_path=self.settings["cert_path"],
            cert_key_path=self.settings["cert_key_path"],
            ollama_host=self.settings["ollama_host"],
            rate_limit=self.settings["rate_limit"],
            burst=self.settings["burst"],
            scheme=scheme,
            listen_port=self.settings["listen_port"],
            use_api_key=self.settings["use_api_key"],
            web_root=str(out_dir) if scheme == SCHEME_LAN else "",
        )
        port = self.settings["listen_port"]
        if scheme == SCHEME_TUNNEL:
            hint = ("nginx.conf записан: {path} — Nginx будет слушать "
                    "127.0.0.1:{port}; далее настройте cloudflared "
                    "(docs/remote_access.md) на hostname {domain} → "
                    "http://127.0.0.1:{port}")
            self._set_status(hint.format(path=path, port=port,
                                         domain=domain))
        elif scheme == SCHEME_LAN:
            # Страница чата: Nginx отдаёт её по / (конфиг уже ссылается)
            page = write_chat_page(out_dir / "chat.html")
            self._set_status(
                f"nginx.conf записан: {path}; страница чата: {page} — "
                f"с телефона откройте http://<IP-ПК>:{port}/ в браузере "
                "(порт откройте в брандмауэре Windows только для частных "
                "сетей; docs/remote_access.md, схема LAN)")
        else:
            hint = ("nginx.conf записан: {path} — положите его в каталог "
                    "Nginx и выполните «nginx -s reload» (подробности: "
                    "docs/remote_access.md)")
            self._set_status(hint.format(path=path, port=port,
                                         domain=domain))
        self.update()

    # ------------------------------------------------------------------
    # Async-обёртки для page.run_task (тяжёлые операции — вне UI-потока)
    # ------------------------------------------------------------------
    async def _check_async(self) -> None:
        try:
            await asyncio.to_thread(self._check_connection)
        except Exception as exc:  # noqa: BLE001 — гард уже страхует
            self._set_status(f"Ошибка проверки: {exc}", error=True)
            self.update()

    async def _update_ddns_async(self) -> None:
        try:
            await asyncio.to_thread(self._update_ddns_now)
        except Exception as exc:  # noqa: BLE001 — гард уже страхует
            self._set_status(f"Ошибка DDNS: {exc}", error=True)
            self.update()

    async def _nginx_async(self) -> None:
        try:
            await asyncio.to_thread(self._generate_nginx)
        except Exception as exc:  # noqa: BLE001 — гард уже страхует
            self._set_status(f"Ошибка генерации nginx.conf: {exc}",
                             error=True)
            self.update()

    # ------------------------------------------------------------------
    # Служебное
    # ------------------------------------------------------------------
    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_text.value = text
        self.status_text.color = (
            ft.Colors.RED_600 if error else ft.Colors.BLUE_GREY_500
        )

    def _schedule(self, handler, *args) -> None:
        """Запускает async-задачу: КОРУТИННУЮ ФУНКЦИЮ в page.run_task
        (готовая корутина даёт TypeError и error-overlay Flet). Без
        page (офлайн-тесты) — no-op: вызывающий код запускает сам."""
        page = getattr(self.app, "page", None)
        if page is None:
            return
        page.run_task(components.guarded_task(self.app, handler), *args)

    def update(self) -> None:
        page = getattr(self.app, "page", None)
        if page is not None:
            page.update()
