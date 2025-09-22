import asyncio
import logging
from typing import Dict, List, Optional
from pymodbus.client import ModbusSerialClient
from pymodbus.exceptions import ModbusException

class VLTDrive:
    def __init__(self, unit_id: int, name: str = None):
        self.unit_id = unit_id
        self.name = name or f"Drive_{unit_id}"
        self.speed_setpoint = 0.0
        self.actual_speed = 0.0
        self.status = "Unknown"
        self.fault = False

    def __repr__(self):
        return f"VLTDrive(unit_id={self.unit_id}, name='{self.name}', speed={self.actual_speed:.1f}%)"

class VLTDriveController:
    def __init__(self, port: str, baudrate: int = 9600, timeout: int = 3, unit_ids: List[int] = None):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.client = None
        self.connected = False
        self.drives: Dict[int, VLTDrive] = {}
        self.logger = logging.getLogger(__name__)

        # Use provided unit IDs or default to 1-6
        self.unit_ids = unit_ids or [1, 2, 3, 4, 5, 6]

        # Initialize drives with configured unit IDs
        for i, unit_id in enumerate(self.unit_ids, 1):
            self.drives[unit_id] = VLTDrive(unit_id=unit_id, name=f"VLT Drive {i} (ID:{unit_id})")

    async def connect(self) -> bool:
        try:
            self.client = ModbusSerialClient(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity='E',  # Even parity (common for VLT)
                stopbits=1,
                bytesize=8
            )

            # Run connection in thread pool
            connected = await asyncio.to_thread(self.client.connect)

            if connected:
                self.connected = True
                self.logger.info(f"Connected to VLT drives on {self.port}")

                # Test communication with each drive
                for drive_id in self.drives.keys():
                    try:
                        await self.read_drive_status(drive_id)
                        self.logger.info(f"VLT Drive {drive_id} responding")
                    except Exception as e:
                        self.logger.warning(f"VLT Drive {drive_id} not responding: {e}")

                return True
            else:
                self.logger.error(f"Failed to connect to {self.port}")
                return False

        except Exception as e:
            self.logger.error(f"Error connecting to VLT drives: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        if self.client:
            try:
                await asyncio.to_thread(self.client.close)
                self.connected = False
                self.logger.info("Disconnected from VLT drives")
            except Exception as e:
                self.logger.error(f"Error disconnecting from VLT drives: {e}")

    async def set_drive_speed(self, drive_id: int, speed_percent: float) -> bool:
        if not self.connected or not self.client:
            raise ConnectionError("VLT drives not connected")

        if drive_id not in self.drives:
            raise ValueError(f"Invalid drive ID: {drive_id}")

        try:
            # Convert percentage to VLT speed reference (typically 0-16383 for 0-100%)
            speed_value = int((speed_percent / 100.0) * 16383)
            speed_value = max(0, min(16383, speed_value))

            # Write to speed reference register (typically register 1)
            result = await asyncio.to_thread(
                self.client.write_register,
                address=1,
                value=speed_value,
                unit=drive_id
            )

            if result.isError():
                self.logger.error(f"Error setting speed for drive {drive_id}: {result}")
                return False

            self.drives[drive_id].speed_setpoint = speed_percent
            self.logger.info(f"Set drive {drive_id} speed to {speed_percent:.1f}%")
            return True

        except Exception as e:
            self.logger.error(f"Error setting speed for drive {drive_id}: {e}")
            return False

    async def read_drive_status(self, drive_id: int) -> VLTDrive:
        if not self.connected or not self.client:
            raise ConnectionError("VLT drives not connected")

        if drive_id not in self.drives:
            raise ValueError(f"Invalid drive ID: {drive_id}")

        try:
            drive = self.drives[drive_id]

            # Read actual speed (typically register 102)
            speed_result = await asyncio.to_thread(
                self.client.read_holding_registers,
                address=102,
                count=1,
                unit=drive_id
            )

            if not speed_result.isError():
                # Convert from VLT units to percentage
                speed_raw = speed_result.registers[0]
                drive.actual_speed = (speed_raw / 16383.0) * 100.0

            # Read status word (typically register 3)
            status_result = await asyncio.to_thread(
                self.client.read_holding_registers,
                address=3,
                count=1,
                unit=drive_id
            )

            if not status_result.isError():
                status_word = status_result.registers[0]

                # Parse status bits (VLT specific)
                if status_word & 0x0001:  # Ready
                    if status_word & 0x0002:  # Running
                        drive.status = "Running"
                    else:
                        drive.status = "Ready"
                else:
                    drive.status = "Not Ready"

                drive.fault = bool(status_word & 0x0008)  # Fault bit

            # Read speed setpoint (typically register 1)
            setpoint_result = await asyncio.to_thread(
                self.client.read_holding_registers,
                address=1,
                count=1,
                unit=drive_id
            )

            if not setpoint_result.isError():
                setpoint_raw = setpoint_result.registers[0]
                drive.speed_setpoint = (setpoint_raw / 16383.0) * 100.0

            return drive

        except Exception as e:
            self.logger.error(f"Error reading status for drive {drive_id}: {e}")
            raise

    async def read_all_drives(self) -> Dict[int, VLTDrive]:
        results = {}
        for drive_id in self.drives.keys():
            try:
                results[drive_id] = await self.read_drive_status(drive_id)
            except Exception as e:
                self.logger.error(f"Failed to read drive {drive_id}: {e}")
                # Return cached drive data
                results[drive_id] = self.drives[drive_id]
        return results

    async def stop_drive(self, drive_id: int) -> bool:
        return await self.set_drive_speed(drive_id, 0.0)

    async def start_drive(self, drive_id: int, speed_percent: float) -> bool:
        # First send start command (typically register 0, bit 0)
        try:
            await asyncio.to_thread(
                self.client.write_register,
                address=0,
                value=0x0001,  # Start command
                unit=drive_id
            )

            # Then set speed
            return await self.set_drive_speed(drive_id, speed_percent)
        except Exception as e:
            self.logger.error(f"Error starting drive {drive_id}: {e}")
            return False

    async def stop_all_drives(self):
        tasks = []
        for drive_id in self.drives.keys():
            tasks.append(self.stop_drive(drive_id))
        await asyncio.gather(*tasks, return_exceptions=True)

    def get_drive(self, drive_id: int) -> Optional[VLTDrive]:
        return self.drives.get(drive_id)

    def is_connected(self) -> bool:
        return self.connected

    def configure_unit_ids(self, unit_ids: List[int]):
        """Reconfigure the unit IDs for VLT drives (requires reconnection to take effect)"""
        self.unit_ids = unit_ids.copy()

        # Clear existing drives
        self.drives.clear()

        # Initialize drives with new unit IDs
        for i, unit_id in enumerate(self.unit_ids, 1):
            self.drives[unit_id] = VLTDrive(unit_id=unit_id, name=f"VLT Drive {i} (ID:{unit_id})")

        self.logger.info(f"Configured VLT drives with unit IDs: {self.unit_ids}")

    def get_unit_ids(self) -> List[int]:
        """Get current unit ID configuration"""
        return self.unit_ids.copy()

    def get_drive_count(self) -> int:
        """Get number of configured drives"""
        return len(self.unit_ids)