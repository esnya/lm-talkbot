"""Neos Connector component"""

import time

import zmq
from websockets.server import WebSocketServerProtocol, serve

from talkbot.utilities.message import is_state_message

from .utilities.config import Config
from .utilities.constants import ComponentState
from .utilities.socket import get_read_socket


async def neos_connector(config: Config = Config()):
    """Connects to Neos and sends messages to the chat engine."""

    logger = config.get_logger("NeosConnector")
    logger.info("Initializing")

    ws_connections: set[WebSocketServerProtocol] = set()

    async def _handle_socket(websocket: WebSocketServerProtocol):
        await websocket.send("ready")
        ws_connections.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            ws_connections.remove(websocket)

    async def _ws_broadcast(message: str):
        logger.info("Expression: %s", message)
        for connection in ws_connections:
            await connection.send(message)

    component_status: dict[str, ComponentState] = {}
    component_status_update_time: dict[str, float] = {}

    def is_busy(component: str):
        state = component_status.get(component, ComponentState.READY)
        time_passed = component and time.time() - component_status_update_time.get(component, 0)
        state_timeout = config.get("neos_connector.expressions.state_timeout", 30)
        return state == ComponentState.BUSY and time_passed < state_timeout

    def get_expression():
        on_busy = config.get("neos_connector.expressions.on_busy", {})
        for component, expression in on_busy.items():
            if is_busy(component):
                return expression
        return config.get("neos_connector.expressions.default", required=True)

    async with serve(
        _handle_socket,
        config.get("neos_connector.websocket.host"),
        config.get("neos_connector.websocket.port"),
    ) as server:
        with get_read_socket(config, config.get("neos_connector.max_interval")) as read_socket:
            logger.info("Initialized")
            logger.info("Started")
            while not read_socket.closed and server.is_serving():
                await _ws_broadcast(get_expression())

                try:
                    message = await read_socket.recv_json()
                    if not is_state_message(message):
                        continue

                    component = message.get("component", "unknown")
                    state = ComponentState(message.get("state")) or ComponentState.READY
                    component_status[component] = state
                    component_status_update_time[component] = time.time()

                    logger.debug(
                        "Status Updated: %s, %s",
                        component_status,
                        component_status_update_time,
                    )
                except zmq.error.Again:
                    pass

        logger.info("Terminated")
