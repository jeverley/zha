"""Support for Zigbee Home Automation covers."""

from __future__ import annotations

import asyncio
from collections import deque
import functools
import logging
from typing import TYPE_CHECKING, Any, cast

from zigpy.zcl.clusters.general import OnOff
from zigpy.zcl.foundation import Status

from zha.application import Platform
from zha.application.platforms import PlatformEntity
from zha.application.platforms.cover.const import (
    ATTR_CURRENT_POSITION,
    ATTR_POSITION,
    ATTR_TILT_POSITION,
    POSITION_CLOSED,
    POSITION_OPEN,
    WCT,
    ZCL_TO_COVER_DEVICE_CLASS,
    CoverDeviceClass,
    CoverEntityFeature,
    CoverState,
    WCAttrs,
)
from zha.application.registries import PLATFORM_ENTITIES
from zha.exceptions import ZHAException
from zha.zigbee.cluster_handlers import ClusterAttributeUpdatedEvent
from zha.zigbee.cluster_handlers.closures import WindowCoveringClusterHandler
from zha.zigbee.cluster_handlers.const import (
    CLUSTER_HANDLER_ATTRIBUTE_UPDATED,
    CLUSTER_HANDLER_COVER,
    CLUSTER_HANDLER_LEVEL,
    CLUSTER_HANDLER_LEVEL_CHANGED,
    CLUSTER_HANDLER_ON_OFF,
    CLUSTER_HANDLER_SHADE,
)
from zha.zigbee.cluster_handlers.general import LevelChangeEvent

if TYPE_CHECKING:
    from zha.zigbee.cluster_handlers import ClusterHandler
    from zha.zigbee.device import Device
    from zha.zigbee.endpoint import Endpoint

_LOGGER = logging.getLogger(__name__)

MULTI_MATCH = functools.partial(PLATFORM_ENTITIES.multipass_match, Platform.COVER)

# Some devices do not stop on the exact target percentage
POSITION_TOLERANCE: int = 1

# Timeout for device movement following a position attribute update
DEFAULT_MOVEMENT_TIMEOUT: float = 5

# Upper limit for dynamic timeout
LIFT_MOVEMENT_TIMEOUT_RANGE: float = 300
TILT_MOVEMENT_TIMEOUT_RANGE: float = 30


@MULTI_MATCH(cluster_handler_names=CLUSTER_HANDLER_COVER)
class Cover(PlatformEntity):
    """Representation of a ZHA cover."""

    PLATFORM = Platform.COVER

    _attr_translation_key: str = "cover"

    def __init__(
        self,
        unique_id: str,
        cluster_handlers: list[ClusterHandler],
        endpoint: Endpoint,
        device: Device,
        **kwargs,
    ) -> None:
        """Init this cover."""
        super().__init__(unique_id, cluster_handlers, endpoint, device, **kwargs)
        cluster_handler = self.cluster_handlers.get(CLUSTER_HANDLER_COVER)
        assert cluster_handler

        self._cover_cluster_handler: WindowCoveringClusterHandler = cast(
            WindowCoveringClusterHandler, cluster_handler
        )
        if self._cover_cluster_handler.window_covering_type:
            self._attr_device_class: CoverDeviceClass | None = (
                ZCL_TO_COVER_DEVICE_CLASS.get(
                    self._cover_cluster_handler.window_covering_type
                )
            )
        self._attr_supported_features: CoverEntityFeature = (
            self._determine_supported_features()
        )

        self._target_lift_position: int | None = None
        self._target_tilt_position: int | None = None
        self._lift_update_received: bool | None = None
        self._tilt_update_received: bool | None = None
        self._lift_state: CoverState | None = None
        self._tilt_state: CoverState | None = None
        self._lift_position_history: deque[int | None] = deque(
            (self.current_cover_position,), maxlen=2
        )
        self._tilt_position_history: deque[int | None] = deque(
            (self.current_cover_tilt_position,), maxlen=2
        )
        self._loop = asyncio.get_running_loop()
        self._movement_timer: asyncio.TimerHandle | None = None

        self._state: CoverState | None = CoverState.OPEN
        self._determine_state(refresh=True)
        self._cover_cluster_handler.on_event(
            CLUSTER_HANDLER_ATTRIBUTE_UPDATED,
            self.handle_cluster_handler_attribute_updated,
        )

    def restore_external_state_attributes(
        self,
        *,
        state: CoverState | None,
        target_lift_position: int | None = None,
        target_tilt_position: int | None = None,
    ):
        """Restore external state attributes."""
        self._state = state
        # Target positions have been removed

    @property
    def supported_features(self) -> CoverEntityFeature:
        """Return supported features."""
        return self._attr_supported_features

    @property
    def state(self) -> dict[str, Any]:
        """Get the state of the cover."""
        response = super().state
        response.update(
            {
                ATTR_CURRENT_POSITION: self.current_cover_position,
                "state": self._state,
                "is_opening": self.is_opening,
                "is_closing": self.is_closing,
                "is_closed": self.is_closed,
            }
        )
        return response

    @property
    def is_closed(self) -> bool | None:
        """Return True if the cover is closed."""
        if self.current_cover_position is None:
            return None
        return self.current_cover_position == POSITION_CLOSED

    @property
    def is_opening(self) -> bool:
        """Return if the cover is opening or not."""
        return self._state == CoverState.OPENING

    @property
    def is_closing(self) -> bool:
        """Return if the cover is closing or not."""
        return self._state == CoverState.CLOSING

    @property
    def current_cover_position(self) -> int | None:
        """Return the current position of ZHA cover.

        In HA None is unknown, 0 is closed, 100 is fully open.
        In ZCL 0 is fully open, 100 is fully closed.
        Keep in mind the values have already been flipped to match HA
        in the WindowCovering cluster handler
        """
        return self._cover_cluster_handler.current_position_lift_percentage

    @property
    def current_cover_tilt_position(self) -> int | None:
        """Return the current tilt position of the cover."""
        return self._cover_cluster_handler.current_position_tilt_percentage

    def _determine_supported_features(self) -> CoverEntityFeature:
        """Determine the supported cover features."""
        supported_features: CoverEntityFeature = (
            CoverEntityFeature.OPEN
            | CoverEntityFeature.CLOSE
            | CoverEntityFeature.STOP
            | CoverEntityFeature.SET_POSITION
        )
        if (
            self._cover_cluster_handler.window_covering_type
            and self._cover_cluster_handler.window_covering_type
            in (
                WCT.Shutter,
                WCT.Tilt_blind_tilt_only,
                WCT.Tilt_blind_tilt_and_lift,
            )
        ):
            supported_features |= CoverEntityFeature.SET_TILT_POSITION
            supported_features |= CoverEntityFeature.OPEN_TILT
            supported_features |= CoverEntityFeature.CLOSE_TILT
            supported_features |= CoverEntityFeature.STOP_TILT
        return supported_features

    @staticmethod
    def _determine_axis_state(
        current: int | None,
        target: int | None,
        history: deque[int | None],
        is_update: bool = False,
    ):
        """Determine cover axis state (lift/tilt).

        Some device update position during movement, others only after stopping.
        When a target is defined the logic aims to mitigate split-brain scenarios
        where a HA command is interrupted by a device button press/physical obstruction.

        The logic considers previous position to determine if the cover is moving,
        if the position has not changed between two device updates it is not moving.
        """

        if current is None:
            return None
        previous = history[0] if history[0] is not None else current

        if target is None and is_update and previous != current:
            target = POSITION_OPEN if current > previous else POSITION_CLOSED

        if (
            target is not None
            and current != target
            and (not is_update or previous != current or history[0] is None)
            and (
                previous <= current < target - POSITION_TOLERANCE
                or target + POSITION_TOLERANCE < current <= previous
            )
        ):
            # ZHA thinks the cover is moving
            return CoverState.OPENING if target > current else CoverState.CLOSING

        # Return the static position
        return CoverState.OPEN if current > POSITION_CLOSED else CoverState.CLOSED

    def _determine_state(
        self,
        is_lift_update: bool = False,
        is_tilt_update: bool = False,
        refresh: bool = False,
    ) -> None:
        """Determine the state of the cover entity.

        This considers current state of both the lift and tilt axis.
        """
        if self._lift_state is None or is_lift_update or refresh:
            self._lift_state = self._determine_axis_state(
                self.current_cover_position,
                self._target_lift_position,
                self._lift_position_history,
                is_lift_update,
            )
        if self._tilt_state is None or is_tilt_update or refresh:
            self._tilt_state = self._determine_axis_state(
                self.current_cover_tilt_position,
                self._target_tilt_position,
                self._tilt_position_history,
                is_tilt_update,
            )

        _LOGGER.debug(
            "_determine_state: lift=(state: %s, is_update: %s, current: %s, target: %s, history: %s), tilt=(state: %s, is_update: %s, current: %s, target: %s, history: %s)",
            self._lift_state,
            is_lift_update,
            self.current_cover_position,
            self._target_lift_position,
            self._lift_position_history,
            self._tilt_state,
            is_tilt_update,
            self.current_cover_tilt_position,
            self._target_tilt_position,
            self._tilt_position_history,
        )

        # Clear target position if the cover axis is not moving
        if self._lift_state not in (CoverState.OPENING, CoverState.CLOSING):
            self._track_target_lift_position(None)
        if self._tilt_state not in (CoverState.OPENING, CoverState.CLOSING):
            self._track_target_tilt_position(None)

        # Start a movement timeout if the cover is moving, else cancel it
        if CoverState.CLOSING in (
            self._lift_state,
            self._tilt_state,
        ) or CoverState.OPENING in (
            self._lift_state,
            self._tilt_state,
        ):
            self._start_movement_timer()
        else:
            self._cancel_movement_timer()

        # Keep the last movement direction if either axis is still moving
        if (
            self.is_closing
            and CoverState.CLOSING in (self._lift_state, self._tilt_state)
            or self.is_opening
            and CoverState.OPENING in (self._lift_state, self._tilt_state)
        ):
            return

        # An open tilt state overrides a closed lift state
        if (
            self._tilt_state == CoverState.OPEN
            and self._lift_state == CoverState.CLOSED
        ):
            self._state = CoverState.OPEN
            return

        # Pick lift state in preference over tilt
        self._state = self._lift_state or self._tilt_state

    def _dynamic_timeout(self) -> float:
        """Return a timer duration in seconds based on expected movement distance.

        This is required because some devices only report position updates after stopping.
        """

        lift_timeout = 0.0
        tilt_timeout = 0.0

        # Calculate dynamic timeout durations if a target is defined and the device has not reported a new position
        if (
            self._target_lift_position is not None
            and self.current_cover_position is not None
            and not self._lift_update_received
        ):
            lift_timeout = (
                abs(self._target_lift_position - self.current_cover_position)
                * 0.01
                * LIFT_MOVEMENT_TIMEOUT_RANGE
            )
        if (
            self._target_tilt_position is not None
            and self.current_cover_tilt_position is not None
            and not self._tilt_update_received
        ):
            tilt_timeout = (
                abs(self._target_tilt_position - self.current_cover_tilt_position)
                * 0.01
                * TILT_MOVEMENT_TIMEOUT_RANGE
            )

        _LOGGER.debug(
            "_dynamic_timeout: lift=(timeout: %s, current: %s, target: %s, update_received: %s), tilt=(timeout: %s, current: %s, target: %s, update_received: %s)",
            lift_timeout,
            self.current_cover_position,
            self._target_lift_position,
            self._lift_update_received,
            tilt_timeout,
            self.current_cover_tilt_position,
            self._target_tilt_position,
            self._tilt_update_received,
        )

        # Return the longest axis movement timeout
        return max(lift_timeout, tilt_timeout)

    def _start_movement_timer(self, seconds: float = 0) -> None:
        """Start timer for clearing the current movement state."""
        if self._movement_timer:
            self._movement_timer.cancel()
        duration = seconds or self._dynamic_timeout() or DEFAULT_MOVEMENT_TIMEOUT
        if duration <= 0:
            raise ZHAException(f"Invalid movement timer duration: {duration}")
        _LOGGER.debug("Movement timer started with a duration of %s seconds", duration)
        self._movement_timer = self._loop.call_later(
            duration, self._clear_movement_state, duration
        )

    def _cancel_movement_timer(self) -> None:
        """Cancel the current movement timer."""
        _LOGGER.debug("Movement timer cancelled")
        if self._movement_timer:
            self._movement_timer.cancel()
            self._movement_timer = None

    def _clear_movement_state(self, duration: float, _=None) -> None:
        """Clear the moving state of the cover due to inactivity."""
        _LOGGER.debug("No movement reported for %s seconds", duration)
        self._target_lift_position = None
        self._target_tilt_position = None
        self._determine_state(refresh=True)
        self.maybe_emit_state_changed_event()

    def _track_target_lift_position(self, position: int | None):
        """Track locally instigated lift movement."""
        self._target_lift_position = position
        if position is not None:
            self._lift_update_received = False
            self._lift_state = None

    def _track_target_tilt_position(self, position: int | None):
        """Track locally instigated tilt movement."""
        self._target_tilt_position = position
        if position is not None:
            self._tilt_update_received = False
            self._tilt_state = None

    @staticmethod
    def _invert_position_for_zcl(position: int) -> int:
        """Convert the HA position to the ZCL position range.

        In HA None is unknown, 0 is closed, 100 is fully open.
        In ZCL 0 is fully open, 100 is fully closed.
        """
        return 100 - position

    def handle_cluster_handler_attribute_updated(
        self, event: ClusterAttributeUpdatedEvent
    ) -> None:
        """Handle position updates from cluster handler.

        The previous position is retained for use in state determination.
        """
        _LOGGER.debug("handle_cluster_handler_attribute_updated=%s", event)
        if event.attribute_id == WCAttrs.current_position_lift_percentage.id:
            self._lift_position_history.append(self.current_cover_position)
            self._lift_update_received = True
            self._determine_state(is_lift_update=True)
        if event.attribute_id == WCAttrs.current_position_tilt_percentage.id:
            self._tilt_position_history.append(self.current_cover_tilt_position)
            self._tilt_update_received = True
            self._determine_state(is_tilt_update=True)
        self.maybe_emit_state_changed_event()

    def async_update_state(self, state):
        """Handle state update from HA operations below."""
        _LOGGER.debug("async_update_state=%s", state)
        self._state = state
        self.maybe_emit_state_changed_event()
        if state in (CoverState.OPENING, CoverState.CLOSING):
            self._start_movement_timer()

    async def async_open_cover(self, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        """Open the cover."""
        self._track_target_lift_position(POSITION_OPEN)
        res = await self._cover_cluster_handler.up_open()
        if res[1] is not Status.SUCCESS:
            self._track_target_lift_position(None)
            raise ZHAException(f"Failed to open cover: {res[1]}")
        self.async_update_state(CoverState.OPENING)

    async def async_open_cover_tilt(self, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        """Open the cover tilt."""
        self._track_target_tilt_position(POSITION_OPEN)
        res = await self._cover_cluster_handler.go_to_tilt_percentage(
            self._invert_position_for_zcl(POSITION_OPEN)
        )
        if res[1] is not Status.SUCCESS:
            self._track_target_tilt_position(None)
            raise ZHAException(f"Failed to open cover tilt: {res[1]}")
        self.async_update_state(CoverState.OPENING)

    async def async_close_cover(self, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        """Close the cover."""
        self._track_target_lift_position(POSITION_CLOSED)
        res = await self._cover_cluster_handler.down_close()
        if res[1] is not Status.SUCCESS:
            self._track_target_lift_position(None)
            raise ZHAException(f"Failed to close cover: {res[1]}")
        self.async_update_state(CoverState.CLOSING)

    async def async_close_cover_tilt(self, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        """Close the cover tilt."""
        self._track_target_tilt_position(POSITION_CLOSED)
        res = await self._cover_cluster_handler.go_to_tilt_percentage(
            self._invert_position_for_zcl(POSITION_CLOSED)
        )
        if res[1] is not Status.SUCCESS:
            self._track_target_tilt_position(None)
            raise ZHAException(f"Failed to close cover tilt: {res[1]}")
        self.async_update_state(CoverState.CLOSING)

    async def async_set_cover_position(self, **kwargs: Any) -> None:
        """Move the cover to a specific position."""
        assert self.current_cover_position is not None
        target_position = kwargs[ATTR_POSITION]
        assert target_position is not None
        self._track_target_lift_position(target_position)
        res = await self._cover_cluster_handler.go_to_lift_percentage(
            self._invert_position_for_zcl(target_position)
        )
        if res[1] is not Status.SUCCESS:
            self._track_target_lift_position(None)
            raise ZHAException(f"Failed to set cover position: {res[1]}")
        self.async_update_state(
            CoverState.CLOSING
            if target_position < self.current_cover_position
            else CoverState.OPENING
        )

    async def async_set_cover_tilt_position(self, **kwargs: Any) -> None:
        """Move the cover tilt to a specific position."""
        assert self.current_cover_tilt_position is not None
        target_position = kwargs[ATTR_TILT_POSITION]
        assert target_position is not None
        self._track_target_tilt_position(target_position)
        res = await self._cover_cluster_handler.go_to_tilt_percentage(
            self._invert_position_for_zcl(target_position)
        )
        if res[1] is not Status.SUCCESS:
            self._track_target_tilt_position(None)
            raise ZHAException(f"Failed to set cover tilt position: {res[1]}")
        self.async_update_state(
            CoverState.CLOSING
            if target_position < self.current_cover_tilt_position
            else CoverState.OPENING
        )

    async def async_stop_cover(self, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        """Stop the cover."""
        self._track_target_lift_position(None)
        res = await self._cover_cluster_handler.stop()
        if res[1] is not Status.SUCCESS:
            raise ZHAException(f"Failed to stop cover: {res[1]}")
        self._determine_state(refresh=True)
        self.maybe_emit_state_changed_event()

    async def async_stop_cover_tilt(self, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        """Stop the cover tilt."""
        self._track_target_tilt_position(None)
        res = await self._cover_cluster_handler.stop()
        if res[1] is not Status.SUCCESS:
            raise ZHAException(f"Failed to stop cover: {res[1]}")
        self._determine_state(refresh=True)
        self.maybe_emit_state_changed_event()


@MULTI_MATCH(
    cluster_handler_names={
        CLUSTER_HANDLER_LEVEL,
        CLUSTER_HANDLER_ON_OFF,
        CLUSTER_HANDLER_SHADE,
    }
)
class Shade(PlatformEntity):
    """ZHA Shade."""

    PLATFORM = Platform.COVER

    _attr_device_class = CoverDeviceClass.SHADE
    _attr_translation_key: str = "shade"

    def __init__(
        self,
        unique_id: str,
        cluster_handlers: list[ClusterHandler],
        endpoint: Endpoint,
        device: Device,
        **kwargs,
    ) -> None:
        """Initialize the ZHA shade."""
        super().__init__(unique_id, cluster_handlers, endpoint, device, **kwargs)
        self._on_off_cluster_handler: ClusterHandler = self.cluster_handlers[
            CLUSTER_HANDLER_ON_OFF
        ]
        self._level_cluster_handler: ClusterHandler = self.cluster_handlers[
            CLUSTER_HANDLER_LEVEL
        ]
        self._is_open: bool = bool(self._on_off_cluster_handler.on_off)
        position = self._level_cluster_handler.current_level
        if position is not None:
            position = max(0, min(255, position))
            position = int(position * 100 / 255)
        self._position: int | None = position
        self._on_off_cluster_handler.on_event(
            CLUSTER_HANDLER_ATTRIBUTE_UPDATED,
            self.handle_cluster_handler_attribute_updated,
        )
        self._level_cluster_handler.on_event(
            CLUSTER_HANDLER_LEVEL_CHANGED, self.handle_cluster_handler_set_level
        )
        self._attr_supported_features: CoverEntityFeature = (
            CoverEntityFeature.OPEN
            | CoverEntityFeature.CLOSE
            | CoverEntityFeature.STOP
            | CoverEntityFeature.SET_POSITION
        )

    @property
    def state(self) -> dict[str, Any]:
        """Get the state of the cover."""
        if (closed := self.is_closed) is None:
            state = None
        else:
            state = CoverState.CLOSED if closed else CoverState.OPEN
        response = super().state
        response.update(
            {
                ATTR_CURRENT_POSITION: self.current_cover_position,
                "is_closed": self.is_closed,
                "state": state,
            }
        )
        return response

    @functools.cached_property
    def is_opening(self) -> bool:
        """Return if the cover is opening or not."""
        return False

    @functools.cached_property
    def is_closing(self) -> bool:
        """Return if the cover is closing or not."""
        return False

    @functools.cached_property
    def supported_features(self) -> CoverEntityFeature:
        """Return supported features."""
        return self._attr_supported_features

    @property
    def current_cover_position(self) -> int | None:
        """Return current position of cover.

        None is unknown, 0 is closed, 100 is fully open.
        """
        return self._position

    @property
    def current_cover_tilt_position(self) -> int | None:
        """Return the current tilt position of the cover."""
        return None

    @property
    def is_closed(self) -> bool | None:
        """Return True if shade is closed."""
        return not self._is_open

    def handle_cluster_handler_attribute_updated(
        self, event: ClusterAttributeUpdatedEvent
    ) -> None:
        """Set open/closed state."""
        if event.attribute_id == OnOff.AttributeDefs.on_off.id:
            self._is_open = bool(event.attribute_value)
            self.maybe_emit_state_changed_event()

    def handle_cluster_handler_set_level(self, event: LevelChangeEvent) -> None:
        """Set the reported position."""
        value = max(0, min(255, event.level))
        self._position = int(value * 100 / 255)
        self.maybe_emit_state_changed_event()

    async def async_open_cover(self, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        """Open the window cover."""
        res = await self._on_off_cluster_handler.on()
        if res[1] != Status.SUCCESS:
            raise ZHAException(f"Failed to open cover: {res[1]}")

        self._is_open = True
        self.maybe_emit_state_changed_event()

    async def async_close_cover(self, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        """Close the window cover."""
        res = await self._on_off_cluster_handler.off()
        if res[1] != Status.SUCCESS:
            raise ZHAException(f"Failed to close cover: {res[1]}")

        self._is_open = False
        self.maybe_emit_state_changed_event()

    async def async_set_cover_position(self, **kwargs: Any) -> None:
        """Move the roller shutter to a specific position."""
        new_pos = kwargs[ATTR_POSITION]
        res = await self._level_cluster_handler.move_to_level_with_on_off(
            new_pos * 255 / 100, 1
        )

        if res[1] != Status.SUCCESS:
            raise ZHAException(f"Failed to set cover position: {res[1]}")

        self._position = new_pos
        self.maybe_emit_state_changed_event()

    async def async_stop_cover(self, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        """Stop the cover."""
        res = await self._level_cluster_handler.stop()
        if res[1] != Status.SUCCESS:
            raise ZHAException(f"Failed to stop cover: {res[1]}")


@MULTI_MATCH(
    cluster_handler_names={CLUSTER_HANDLER_LEVEL, CLUSTER_HANDLER_ON_OFF},
    manufacturers="Keen Home Inc",
)
class KeenVent(Shade):
    """Keen vent cover."""

    _attr_device_class = CoverDeviceClass.DAMPER
    _attr_translation_key: str = "keen_vent"

    async def async_open_cover(self, **kwargs: Any) -> None:
        """Open the cover."""
        position = self._position or 100
        await asyncio.gather(
            self._level_cluster_handler.move_to_level_with_on_off(
                position * 255 / 100, 1
            ),
            self._on_off_cluster_handler.on(),
        )

        self._is_open = True
        self._position = position
        self.maybe_emit_state_changed_event()
