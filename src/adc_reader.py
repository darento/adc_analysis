import struct
import time
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
from typing import Generator, Iterator, Tuple, List

import numpy as np
from typing import List, Tuple
from event_formats import EVENT_DTYPE_V1, EVENT_DTYPE_V2


class EventParser:
    def __init__(self, version: str = "v2"):
        self.version = version.lower()
        if self.version == "v1":
            self.event_dtype = EVENT_DTYPE_V1
        elif self.version == "v2":
            self.event_dtype = EVENT_DTYPE_V2
        else:
            raise ValueError(f"Unsupported version: {version}")

    def parse(self, buffer: bytes, order: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        if self.version == "v1":
            return self._parse_v1(buffer, order)
        elif self.version == "v2":
            return self._parse_v2(buffer, order)

    def _parse_v1(
        self, buffer: bytes, order: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        events = np.frombuffer(buffer, dtype=self.event_dtype)
        num_events = len(events)

        headers = np.zeros(
            num_events,
            dtype=[
                ("CTRL", "u1"),
                ("D3-D0", "u1"),
                ("Timestamp", ">u4"),
                ("S", "u1"),
                ("ModuleID", "u1"),
            ],
        )

        headers["CTRL"] = (events["CTRL"] & 0xF0) >> 4
        headers["D3-D0"] = events["CTRL"] & 0x0F
        headers["Timestamp"] = events["Timestamp"]
        headers["S"] = (events["S_ModuleID"] & 0x80) >> 7
        headers["ModuleID"] = events["S_ModuleID"] & 0x7F

        adc_data = events["ADC_Data"].astype(np.uint16)
        adc_channels = np.zeros((num_events, 33), dtype=np.uint16)

        adc_channels[:, :-1:2] = (
            (adc_data[:, :-2:3] << 4) | (adc_data[:, 1:-1:3] >> 4)
        ) & 0xFFF

        adc_channels[:, 1::2] = (
            ((adc_data[:, 1:-1:3] & 0x0F) << 8) | adc_data[:, 2::3]
        ) & 0xFFF

        adc_channels[:, -1] = ((adc_data[:, -2] << 4) | (adc_data[:, -1] >> 4)) & 0xFFF

        mask = headers["S"] == 0
        return headers["Timestamp"][mask], adc_channels[mask][:, order]

    def _parse_v2(
        self, buffer: bytes, order: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        events = np.frombuffer(buffer, dtype=self.event_dtype)
        num_events = len(events)

        cnt = (events["Header0"] & 0b11000000) >> 6
        timestamp_coarse = (
            ((events["Header0"] & 0x3F).astype(np.uint64) << 32)
            | (events["Timestamp_1"].astype(np.uint64) << 24)
            | (events["Timestamp_2"].astype(np.uint64) << 16)
            | (events["Timestamp_3"].astype(np.uint64) << 8)
            | (events["Timestamp_4"].astype(np.uint64))
        )

        s_flags = (events["Header1"] & 0x80) >> 7
        module_addr = events["Header1"] & 0x7F

        adc_data = events["ADC_Data"].astype(np.uint16)
        adc_channels = np.zeros((num_events, 33), dtype=np.uint16)

        adc_channels[:, :-1:2] = (
            (adc_data[:, :-2:3] << 4) | (adc_data[:, 1:-1:3] >> 4)
        ) & 0xFFF

        adc_channels[:, 1::2] = (
            ((adc_data[:, 1:-1:3] & 0x0F) << 8) | adc_data[:, 2::3]
        ) & 0xFFF

        adc_channels[:, -1] = ((adc_data[:, -2] << 4) | (adc_data[:, -1] >> 4)) & 0xFFF

        d0 = (adc_data[:, -1] & 0x08) >> 3  # bit 3 is D0
        t_msb = adc_data[:, -1] & 0x07  # bits 0-2 are T MSB

        full_timestamp = (t_msb.astype(np.uint64) << 38) | timestamp_coarse

        mask = s_flags == 0
        return full_timestamp[mask], adc_channels[mask][:, order]


def _detect_coincidences(
    timestamps: List[Tuple[int, int]], adc_data_list: List[List[int]], threshold: int
) -> List[Tuple[List[Tuple[int, int]], List[List[int]]]]:
    """
    Detects coincidences in the data.

    Args:
        timestamps (List[Tuple[int, int]]): List of tuples with event index and timestamp.
        adc_data_list (List[List[int]]): List of ADC channel values.
        threshold (int): Threshold for coincidence detection.

    Returns:
        List[Tuple[List[Tuple[int, int]], List[List[int]]]]: List of coincidences.
    """
    coincidences = []
    current_group = [(timestamps[0][0], timestamps[0][1])]
    tstp_group = timestamps[0][1]
    for i in range(1, len(timestamps)):
        delta = timestamps[i][1] - tstp_group
        if delta <= threshold and delta >= 0:
            current_group.append((timestamps[i][0], timestamps[i][1]))
        elif delta < 0:
            if len(current_group) > 1:
                coincident_adc_values = [
                    adc_data_list[j[0]] for j in current_group[0:2]
                ]
                coincidences.append((current_group[0:2], coincident_adc_values))
            current_group = [(timestamps[i][0], timestamps[i][1])]
            tstp_group = timestamps[i][1]
        else:
            if len(current_group) > 1:
                coincident_adc_values = [
                    adc_data_list[j[0]] for j in current_group[0:2]
                ]
                coincidences.append((current_group[0:2], coincident_adc_values))
            current_group = [(timestamps[i][0], timestamps[i][1])]
            tstp_group = timestamps[i][1]
    return coincidences


def _send_coincidences_to_queue(
    coincidences: List[Tuple[List[Tuple[int, int]], List[List[int]]]], queue: Queue
):
    """
    Sends coincidences to the writer process via the queue.

    Args:
        coincidences (List[Tuple[List[Tuple[int, int]], List[List[int]]]]): List of coincidences.
        queue (Queue): Queue to send coincidences to the writer process.
    """
    for group, adc_values in coincidences:
        queue.put((group, adc_values))


def writer_process(output_file: str, queue: Queue):
    """
    Writes coincidences (timestamps and ADC values) to a file.

    Args:
        output_file (str): Path to the output file.
        queue (Queue): Queue to receive coincidences from worker processes.
    """
    batch = []
    with open(output_file, "w") as file:
        while True:
            data = queue.get()
            if data == "DONE":
                break
            batch.append(data)
            # Write the batch when it reaches a certain size
            if len(batch) >= 1000:
                for group, adc_values in batch:
                    for g, adc in zip(group, adc_values):
                        file.write(
                            "\t".join(map(str, g))
                            + "\t"
                            + "\t".join(map(str, adc))
                            + "\n"
                        )
                batch = []  # Clear the batch
        # Write remaining data
        for group, adc_values in batch:
            for g, adc in zip(group, adc_values):
                file.write(
                    "\t".join(map(str, g)) + "\t" + "\t".join(map(str, adc)) + "\n"
                )


def worker_process(
    file_path: str,
    start: int,
    end: int,
    num_ev_buf: int,
    event_size: int,
    order: list,
    threshold: int,
    queue: Queue,
):
    """
    Processes the file and sends coincidences to the queue.

    Args:
        file_path (str): Path to the file.
        start (int): Start byte position.
        end (int): End byte position.
        num_ev_buf (int): Size of the buffer to read.
        event_size (int): Size of each event in bytes.
        order (list): The order of the ADC channels.
        threshold (int): Threshold for coincidence detection.
        queue (Queue): Queue to send coincidences to.
    """
    timestamp_list = []
    cnt_chunks = 0

    for timestamps, adc_data_list in read_file_chunks(
        file_path, start, end, num_ev_buf, event_size, order
    ):
        # Detect coincidences
        coincidences = _detect_coincidences(timestamps, adc_data_list, threshold)

        # Send coincidences to the writer process
        _send_coincidences_to_queue(coincidences, queue)

        timestamp_list.extend(timestamps)
        cnt_chunks += 1


def read_file_chunks(
    file_path: str,
    start: int,
    end: int,
    num_ev_buf: int,
    event_size: int,
    order: list,
    parser: EventParser,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Processes a segment of the file in smaller chunks using EventParser.

    Args:
        file_path (str): Path to the file.
        start (int): Start byte position.
        end (int): End byte position.
        num_ev_buf (int): Size of the buffer to read.
        event_size (int): Size of each event in bytes.
        order (list): The order of the ADC channels.
        parser (EventParser): EventParser instance to use for parsing.
    """
    buffer_size = num_ev_buf * event_size  # Define a buffer size for smaller chunks
    with open(file_path, "rb") as file:
        file.seek(start)
        current_position = start
        with tqdm(
            total=end - start,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Process {start // (end - start)}",
        ) as pbar:
            while current_position < end:
                # Read a smaller chunk from the file
                read_size = min(buffer_size, end - current_position)
                buffer = file.read(read_size)
                current_position += read_size
                pbar.update(read_size)

                # Use EventParser instead of _extract_data
                timestamps, adc_channels = parser.parse(buffer, order)

                yield timestamps, adc_channels


if __name__ == "__main__":
    # Example usage
    # Read the file path from command-line arguments
    # file_path = "P:/Valencia/I3M/Proyectos/DeepBrain/data/05022024_FOV0_0_600s.raw"
    file_path = "..\\..\\..\\data\\DeepBrain\\05022024_FOV0_0_600s.raw"
    output_file = (
        "P:/Valencia/I3M/Proyectos/DeepBrain/data/05022024_FOV0_0_600s_coincidences.txt"
    )
    num_ev_buf = 10000
    event_size = 56
    # order = [1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    # order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    order = list(range(33))
    num_processes = 1
    threshold = 10000
    file_size = os.path.getsize(file_path)
    chunk_size = file_size // num_processes
    timestamp_list = []
    chunk_cnt = 0

    # Initialize EventParser with V1 format
    parser = EventParser(version="v1")

    for timestamps, adc_data_list in read_file_chunks(
        file_path, 0, file_size, num_ev_buf, event_size, order, parser
    ):
        if chunk_cnt <= 1:
            timestamp_list.extend(timestamps)
        chunk_cnt += 1
        pass

    time_list = (
        np.array(timestamp_list) - timestamp_list[0]
    ) / 1e6  # Convert to seconds

    fig = plt.figure(figsize=(12, 6))
    plt.plot(timestamp_list, "o-")
    plt.xlabel("Event Index")
    plt.ylabel("Timestamp (s)")
    plt.title("Timestamps of Events")
    plt.grid()
    plt.tight_layout()

    fig = plt.figure(figsize=(12, 6))
    plt.plot(time_list, "o-")
    plt.xlabel("Event Index")
    plt.ylabel("Time (s)")
    plt.title("Time of Events")
    plt.grid()
    plt.tight_layout()
    plt.show()

    """

    # Create a Queue for communication
    queue = Queue(maxsize=100000)

    # Start writer process
    writer = Process(target=writer_process, args=(output_file, queue))
    writer.start()

    # Start worker processes
    processes = []
    for i in range(num_processes):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_processes - 1 else file_size
        process = Process(
            target=worker_process,
            args=(
                file_path,
                start,
                end,
                num_ev_buf,
                event_size,
                order,
                threshold,
                queue,
            ),
        )
        processes.append(process)
        process.start()

    # Wait for workers to finish
    for process in processes:
        process.join()

    # Signal the writer to stop
    queue.put("DONE")
    writer.join()
    """
