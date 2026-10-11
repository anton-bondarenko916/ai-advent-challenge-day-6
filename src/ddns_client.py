"""DDNS-клиент: актуальный IP на бесплатном домене (ТЗ ai_1_30).

Удалённый доступ к локальной Ollama идёт через домен (например,
``myai.duckdns.org``), который всегда указывает на текущий публичный
IP домашней сети. Пока IP динамический, его нужно периодически
сообщать DDNS-сервису — этим занимается этот клиент:

* :func:`update_duckdns` / :func:`update_aperto_dns` /
  :func:`update_noip` — обновление IP у конкретного провайдера
  (GET/POST, ключ/токен провайдера);
* :class:`DDNSClient` — обёртка над провайдерами: разовое обновление
  (:meth:`DDNSClient.update_once`, ошибки сети не выбрасываются, а
  возвращаются как ``(False, сообщение)``) и бесконечный цикл
  (:meth:`DDNSClient.run_forever`);
* :func:`ddns_loop` — функция цикла из ТЗ (update_func + интервал).

Клиент НЕ запускается внутри приложения: по ТЗ он работает постоянно,
даже когда чат закрыт, — поэтому его запускают как службу Windows или
задачу планировщика (команды — в docs/remote_access.md и epilog CLI)::

    python -m src.ddns_client --provider duckdns --domain myai.duckdns.org \\
        --token XXXX --interval-minutes 5

DuckDNS принимает голый поддомен (``myai``), полный домен
``myai.duckdns.org`` нормализуется автоматически.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Callable, Optional, Tuple

import httpx

logger = logging.getLogger("deepseek-chat.ddns")

# Эндпоинты обновления IP (протоколы провайдеров)
DUCKDNS_UPDATE_URL = "https://www.duckdns.org/update"
APERTO_UPDATE_URL = "https://api.apertodns.com/update"
NOIP_UPDATE_URL = "https://dynupdate.no-ip.com/nic/update"

DDNS_TIMEOUT = 10.0
DEFAULT_INTERVAL_MINUTES = 5

# Ключи провайдеров (реестр RemoteAccessSettings.DDNS_PROVIDERS — подписи)
PROVIDER_DUCKDNS = "duckdns"
PROVIDER_APERTO = "aperto"
PROVIDER_NOIP = "noip"
PROVIDERS = (PROVIDER_DUCKDNS, PROVIDER_APERTO, PROVIDER_NOIP)


def duckdns_domain(domain: str) -> str:
    """Голый поддомен DuckDNS (``myai.duckdns.org`` → ``myai``).

    API DuckDNS ждёт поддомен без суффикса; полный домен тоже
    принимается users-скриптами часто неверно, поэтому нормализуем.
    """
    return str(domain or "").strip().lower().removesuffix(".duckdns.org")


def update_duckdns(domain: str, token: str,
                   timeout: float = DDNS_TIMEOUT) -> bool:
    """Обновляет IP в DuckDNS (пустой ``ip`` — сервис определит сам).

    Args:
        domain: домен или поддомен (``myai.duckdns.org`` / ``myai``).
        token: токен из личного кабинета DuckDNS.
        timeout: таймаут HTTP-запроса.

    Returns:
        True — сервис ответил ``OK`` (IP записан или не изменился).
    """
    response = httpx.get(
        DUCKDNS_UPDATE_URL,
        params={"domains": duckdns_domain(domain), "token": token, "ip": ""},
        timeout=timeout,
    )
    return response.text.strip().upper() == "OK"


def update_aperto_dns(domain: str, api_key: str,
                      timeout: float = DDNS_TIMEOUT) -> bool:
    """Обновляет IP в ApertoDNS (POST + Bearer-ключ)."""
    response = httpx.post(
        APERTO_UPDATE_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"domain": domain},
        timeout=timeout,
    )
    return response.status_code == 200


def update_noip(domain: str, username: str, password: str,
                timeout: float = DDNS_TIMEOUT) -> bool:
    """Обновляет IP в No-IP (протокол nic/update, Basic-аутентификация).

    Успех — ответ ``good`` (IP обновлён) или ``nochg`` (не изменился);
    ``nohost``/``badauth``/``911`` и прочие — отказ.
    """
    response = httpx.get(
        NOIP_UPDATE_URL,
        params={"hostname": domain, "myip": ""},
        auth=(username, password),
        timeout=timeout,
    )
    text = response.text.strip().lower()
    return (response.status_code == 200
            and (text.startswith("good") or text.startswith("nochg")))


def ddns_loop(update_func: Callable[[], Any],
              interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
              max_iterations: Optional[int] = None,
              on_error: Optional[Callable[[Exception], None]] = None) -> None:
    """Цикл обновления DDNS: каждые ``interval_minutes`` минут.

    Args:
        update_func: функция обновления (одна из update_*).
        interval_minutes: пауза между обновлениями (0 — без паузы).
        max_iterations: ограничение числа обновлений (None — вечно);
            для тестов.
        on_error: обработчик ошибки (по умолчанию — журнал).
    """
    handler = on_error or (lambda exc: logger.warning("DDNS error: %s", exc))
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        try:
            update_func()
        except Exception as exc:  # noqa: BLE001 — цикл не должен умирать
            handler(exc)
        iterations += 1
        if max_iterations is not None and iterations >= max_iterations:
            break
        time.sleep(max(0, int(interval_minutes)) * 60)


class DDNSClient:
    """Клиент одного DDNS-провайдера: разовое обновление + вечный цикл.

    Attributes:
        provider: ключ провайдера (duckdns / aperto / noip).
        domain: DDNS-домен (для DuckDNS нормализуется до поддомена).
        token: токен/ключ провайдера; у No-IP — имя пользователя.
        secret: пароль No-IP (остальным провайдерам не нужен).
        interval_minutes: пауза между обновлениями в run_forever.
    """

    def __init__(self, provider: str = PROVIDER_DUCKDNS,
                 domain: str = "", token: str = "", secret: str = "",
                 interval_minutes: int = DEFAULT_INTERVAL_MINUTES) -> None:
        self.provider = str(provider or PROVIDER_DUCKDNS).strip().lower()
        self.domain = str(domain or "").strip()
        self.token = str(token or "").strip()
        self.secret = str(secret or "").strip()
        self.interval_minutes = max(0, int(interval_minutes))

    def update_once(self) -> Tuple[bool, str]:
        """Разовое обновление IP.

        Ошибки сети/сервиса НЕ выбрасываются — возвращаются вторым
        элементом (планировщик задач не должен копить упавшие процессы).

        Returns:
            (успех, человекочитаемое сообщение для UI/журнала).
        """
        try:
            if self.provider == PROVIDER_DUCKDNS:
                ok = update_duckdns(self.domain, self.token)
            elif self.provider == PROVIDER_APERTO:
                ok = update_aperto_dns(self.domain, self.token)
            elif self.provider == PROVIDER_NOIP:
                ok = update_noip(self.domain, self.token, self.secret)
            else:
                return False, (f"Неизвестный DDNS-провайдер: "
                               f"{self.provider}. Доступные: "
                               f"{', '.join(PROVIDERS)}")
        except Exception as exc:  # noqa: BLE001 — сеть/HTTP/таймаут
            return False, f"Ошибка DDNS ({self.provider}): {exc}"
        if ok:
            return True, f"IP обновлён: {self.domain} ({self.provider})"
        return False, (f"{self.provider} отклонил обновление "
                       f"(проверьте домен и токен)")

    def run_forever(self, max_iterations: Optional[int] = None) -> None:
        """Вечный цикл обновления (для службы/планировщика задач)."""
        logger.info(
            "DDNS-клиент запущен: %s, домен %s, интервал %s мин.",
            self.provider, self.domain, self.interval_minutes)

        def _iterate() -> None:
            ok, message = self.update_once()
            (logger.info if ok else logger.warning)(
                "DDNS %s: %s", self.provider, message)

        ddns_loop(_iterate, interval_minutes=self.interval_minutes,
                  max_iterations=max_iterations)


def main(argv: Optional[list] = None) -> int:
    """CLI для службы/планировщика: обновлять IP вечно (Ctrl+C — стоп)."""
    parser = argparse.ArgumentParser(
        description="DDNS-клиент: держит домен на актуальном публичном IP.",
        epilog=(
            "Планировщик задач Windows (запуск каждые 5 минут, одно "
            "обновление за запуск): schtasks /Create /TN \"DDNS Update\" "
            "/SC MINUTE /MO 5 /TR \"python -m src.ddns_client --provider "
            "duckdns --domain myai.duckdns.org --token XXXX "
            "--run-once\"  (рабочий каталог — корень проекта)."
        ))
    parser.add_argument("--provider", default=PROVIDER_DUCKDNS,
                        choices=PROVIDERS, help="DDNS-провайдер")
    parser.add_argument("--domain", required=True,
                        help="DDNS-домен (например, myai.duckdns.org)")
    parser.add_argument("--token", default="",
                        help="токен/ключ провайдера (у No-IP — логин)")
    parser.add_argument("--secret", default="",
                        help="пароль (только No-IP)")
    parser.add_argument("--interval-minutes", type=int,
                        default=DEFAULT_INTERVAL_MINUTES,
                        help="интервал цикла в минутах")
    parser.add_argument("--run-once", action="store_true",
                        help="одно обновление и выход (для планировщика)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    client = DDNSClient(provider=args.provider, domain=args.domain,
                        token=args.token, secret=args.secret,
                        interval_minutes=args.interval_minutes)
    if args.run_once:
        ok, message = client.update_once()
        (logger.info if ok else logger.error)("DDNS: %s", message)
        return 0 if ok else 1
    client.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
