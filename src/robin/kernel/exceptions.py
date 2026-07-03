"""Exceptions for the kernel module."""

from robin.supply.entities import Service


class TicketNotBoughtException(Exception):
    """Raised when the ticket is not bought."""

    def __init__(self, service: Service, *args, **kwargs):
        msg = f"A ticket for the service '{service.id}' was not possible to buy."
        super().__init__(msg, *args, **kwargs)
