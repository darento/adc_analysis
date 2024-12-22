import struct
import time
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
from typing import Generator, Iterator, Tuple, List


# Define the structured dtype for an event as a constant
EVENT_DTYPE = np.dtype(
    [
        ("CTRL", "u1"),  # 1 byte (CTRL and D3-D0 combined)
        ("Timestamp", ">u4"),  # 4 bytes (Tstp big-endian)
        ("S_ModuleID", "u1"),  # 1 byte (S and ModuleID combined)
        ("ADC_Data", "50B"),  # 50 bytes (ADC data + 4 bit unused)
    ]
)


def _extract_data(
    buffer: bytes, event_size: int, order: list
) -> Tuple[List[Tuple[int, int]], List[List[int]]]:
    """
    Extracts timestamps and ADC data from the buffer.

    Args:
        buffer (bytes): The buffer containing the data.
        event_size (int): Size of each event in bytes.
        order (list): The order of the ADC channels.

    Returns:
        Tuple[List[Tuple[int, int]], List[List[int]]]: A tuple containing:
            - List of tuples with event index and timestamp.
            - List of ADC channel values.
    """
    # Load the buffer as a structured array
    events = np.frombuffer(buffer, dtype=EVENT_DTYPE)

    # Extract headers
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

    # Extract ADC channels
    adc_data = events["ADC_Data"].astype(np.uint16)  # Ensure data is uint16
    adc_channels = np.zeros((num_events, 33), dtype=np.uint16)  # Correct dtype

    # Decode 12-bit ADC values
    adc_channels[:, :-1:2] = (
        (adc_data[:, :-2:3].astype(np.uint16) << 4)
        | (adc_data[:, 1:-1:3].astype(np.uint16) >> 4)
    ) & 0xFFF
    adc_channels[:, 1::2] = (
        (adc_data[:, 1:-1:3].astype(np.uint16) & 0x0F) << 8
    ) | adc_data[:, 2::3].astype(np.uint16)
    adc_channels[:, -1] = (
        (adc_data[:, -2].astype(np.uint16) << 4)
        | (adc_data[:, -1].astype(np.uint16) >> 4)
    ) & 0xFFF

    # Retrieve only headers and adc_channels for the events with S=0
    timestamps = headers["Timestamp"][headers["S"] == 0]
    adc_channels = adc_channels[headers["S"] == 0]

    # Retrieve only the columns of the adc_channels specified in the order list
    adc_channels = adc_channels[:, order]

    return timestamps, adc_channels


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
    file_path: str, start: int, end: int, num_ev_buf: int, event_size: int, order: list
) -> Generator[Tuple[List[Tuple[int, int]], List[List[int]]], None, None]:
    """
    Processes a segment of the file in smaller chunks, detects coincidences,
    and sends results to the writer process.

    Args:
        file_path (str): Path to the file.
        start (int): Start byte position.
        end (int): End byte position.
        num_ev_buf (int): Size of the buffer to read.
        event_size (int): Size of each event in bytes.
        order (list): The order of the ADC channels.
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

                headers, adc_channels = _extract_data(buffer, event_size, order)

                yield headers, adc_channels


if __name__ == "__main__":
    # Example usage
    # Read the file path from command-line arguments
    file_path = "P:/Valencia/I3M/Proyectos/DeepBrain/data/05022024_FOV0_0_600s.raw"
    output_file = (
        "P:/Valencia/I3M/Proyectos/DeepBrain/data/05022024_FOV0_0_600s_coincidences.txt"
    )
    num_ev_buf = 10000
    event_size = 56
    # order = [1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    # order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    order = range(33)
    num_processes = 1
    threshold = 10000
    file_size = os.path.getsize(file_path)
    chunk_size = file_size // num_processes
    timestamp_list = []
    chunk_cnt = 0
    for timestamps, adc_data_list in read_file_chunks(
        file_path, 0, file_size, num_ev_buf, event_size, order
    ):
        # print(f"First 10 timestamps: {timestamps[0]}")
        # print(f"First 10 ADC data: {adc_data_list[0]}")
        # input("Press Enter to continue...")
        # for t, adc in zip(timestamps, adc_data_list):
        #     print(t, adc)
        #     input("Press Enter to continue...")
        if chunk_cnt <= 1:
            timestamp_list.extend(timestamps)
        chunk_cnt += 1
        pass

    plt.plot(timestamp_list, "o-")
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
