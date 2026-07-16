"""Security utilities: rate limiting and input sanitization.

There is no API key auth on purpose. This is a reference service meant to run
on one machine: docker-compose binds every published port to the loopback
interface, so nothing is reachable from the network. A shipped default
credential would read as protection while being public knowledge. Put a real
gateway in front of the API before exposing it beyond localhost.
"""

import bleach
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)


def sanitize_question(text: str) -> str:
    """Strip any HTML or script tags from user input before it is used."""
    return bleach.clean(text, tags=[], strip=True).strip()
