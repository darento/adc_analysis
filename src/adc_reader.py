import struct
from typing import Iterator, Tuple, List


def read_raw_file(file_path: str) -> Iterator[Tuple[dict, List[int]]]:
    """
    Reads a RAW ADC binary file and parses its structure.

    Args:
        file_path (str): Path to the RAW ADC binary file.

    Yields:
        Tuple[dict, List[int]]: A tuple containing:
            - Header as a dictionary with `CTRL`, `D3-D0`, `Timestamp`, `S`, and `ModuleID`.
            - A list of 33 ADC channel values (int, 12-bit resolution).
    """
    with open(file_path, "rb") as file:
        while chunk := file.read(56):  # Each block is 56 bytes
            if len(chunk) != 56:
                print("Incomplete data block detected. Skipping...")
                continue

            # Parse the first byte (CTRL and D3-D0)
            ctrl_digital = chunk[0]
            ctrl = (ctrl_digital & 0xF0) >> 4  # Upper 4 bits for CTRL
            digital_signals = ctrl_digital & 0x0F  # Lower 4 bits for D3-D0

            # Parse the timestamp (Bytes 2–5)
            timestamp = struct.unpack(">I", chunk[1:5])[0]  # Big-endian 32-bit integer

            # Parse Byte 6 (S and ModuleID)
            s_module = chunk[5]
            s = (s_module & 0x80) >> 7  # Most significant bit for S
            module_id = s_module & 0x7F  # Remaining 7 bits for ModuleID

            # Parse ADC channels (Bytes 7–55)
            adc_data = chunk[6:]  # Remaining bytes contain ADC data
            adc_channels = []

            # Extract 12-bit ADC values safely
            for i in range(0, len(adc_data) - 2, 3):  # Process 3 bytes at a time
                byte1, byte2, byte3 = struct.unpack("BBB", adc_data[i : i + 3])

                adc1 = ((byte1 << 4) | (byte2 >> 4)) & 0xFFF  # First 12 bits
                adc2 = ((byte2 & 0x0F) << 8) | byte3  # Second 12 bits

                print(
                    f"Bytes: {byte1:02X} {byte2:02X} {byte3:02X} -> ADC1: {adc1:03X}, ADC2: {adc2:03X}"
                )

                adc_channels.append(adc1)
                adc_channels.append(adc2)

            # ADC 33 is stored in the first 12 bits of the last 2 bytes, the last 4 bits are unusued
            byte1, byte2 = struct.unpack("BB", adc_data[-2:])

            adc1 = ((byte1 << 4) | (byte2 >> 4)) & 0xFFF  # First 12 bits
            unused = byte2 & 0x0F  # Last 4 bits

            print(
                f"Bytes: {byte1:02X} {byte2:02X} {byte3:02X} -> ADC1: {adc1:03X}, ADC2: {adc2:03X}"
            )

            adc_channels.append(adc1)

            # Print the length of adc_channels before truncating
            print(f"Length of adc_channels before truncating: {len(adc_channels)}")

            # Only keep the first 33 channels
            adc_channels = adc_channels[:33]

            # Print the length of adc_channels after truncating
            print(f"Length of adc_channels after truncating: {len(adc_channels)}")

            # Yield the parsed data
            yield {
                "CTRL": ctrl,
                "D3-D0": [bool(digital_signals & (1 << i)) for i in range(4)],
                "Timestamp": timestamp,
                "S": s,
                "ModuleID": module_id,
                "ADC Channels": adc_channels,
            }


if __name__ == "__main__":
    pass
