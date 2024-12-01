import struct
import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
from typing import Iterator, Tuple, List


def _parse_header(chunk: bytes) -> dict:
    """
    Parses the header of a data chunk.

    Args:
        chunk (bytes): A 56-byte data chunk.

    Returns:
        dict: A dictionary containing:
            - `CTRL` (int): Control signals.
            - `D3-D0` (int): Digital signals.
            - `Timestamp` (int): Timestamp.
            - `S` (int): Status bit.
            - `ModuleID` (int): Module ID.
    """
    ctrl_digital = chunk[0]
    ctrl = (ctrl_digital & 0xF0) >> 4
    digital_signals = ctrl_digital & 0x0F

    timestamp = struct.unpack(">I", chunk[1:5])[0]

    s_module = chunk[5]
    s = (s_module & 0x80) >> 7
    module_id = s_module & 0x7F

    return {
        "CTRL": ctrl,
        "D3-D0": digital_signals,
        "Timestamp": timestamp,
        "S": s,
        "ModuleID": module_id,
    }

def _parse_adc_channels(adc_data: bytes, order: list) -> List[int]:
    """
    Parses the ADC channel values from a data chunk.

    Args:
        adc_data (bytes): The ADC data part of the chunk (50 bytes).

    Returns:
        List[int]: A list of 33 ADC channel values (int, 12-bit resolution).
    """
    adc_channels = [0] * 33
    bytes_data = struct.unpack(f"{len(adc_data)}B", adc_data)
    for i in range(0, len(adc_data) - 2, 3):
        byte1, byte2, byte3 = bytes_data[i:i + 3]

        adc1 = ((byte1 << 4) | (byte2 >> 4)) & 0xFFF
        adc2 = ((byte2 & 0x0F) << 8) | byte3

        adc_channels[i // 3 * 2] = adc1
        adc_channels[i // 3 * 2 + 1] = adc2

    # Process the last ADC (ADC33)
    byte1, byte2 = bytes_data[-2:]
    adc33 = ((byte1 << 4) | (byte2 >> 4)) & 0xFFF
    adc_channels[-1] = adc33

    # Send only the channels in the order specified
    adc_channels = [adc_channels[i - 1] for i in order]

    return adc_channels


def writer_process(output_file: str, queue: Queue):
    """
    Writes coincidences (timestamps and ADC values) to a file.

    Args:
        output_file (str): Path to the output file.
        queue (Queue): Queue to receive coincidences from worker processes.
    """
    with open(output_file, "w") as file:
        while True:
            data = queue.get()
            if data == "DONE":
                break
            groups, adc_values = data
            # write per columns in the file, where each line is a coincidence
            for group, adc_value in zip(groups, adc_values):
                file.write("\t".join(map(str, group)) + "\t" + "\t".join(map(str, adc_value)) + "\n")


            


def _read_buffer(file, start: int, end: int, buffer_size: int) -> bytes:
    """
    Reads a buffer from the file.

    Args:
        file: File object.
        start (int): Start byte position.
        end (int): End byte position.
        buffer_size (int): Size of the buffer to read.

    Returns:
        bytes: The read buffer.
    """
    file.seek(start)
    buffer = b""
    while start < end:
        read_size = min(buffer_size, end - start)
        buffer += file.read(read_size)
        start += read_size
    return buffer

def _extract_data(buffer: bytes, event_size: int, order: list) -> Tuple[List[Tuple[int, int]], List[List[int]]]:
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
    timestamps = []
    adc_data_list = []
    valid_event_index = 0
    for i in range(0, len(buffer), event_size):
        if i + event_size > len(buffer):
            break
        chunk = buffer[i:i + event_size]
        header = _parse_header(chunk)
        if header["S"] == 1:
            continue
        adc_data = _parse_adc_channels(chunk[6:], order)
        timestamps.append((valid_event_index, header["Timestamp"]))
        adc_data_list.append(adc_data)
        valid_event_index += 1
        
    return timestamps, adc_data_list

def _detect_coincidences(timestamps: List[Tuple[int, int]], adc_data_list: List[List[int]], threshold: int) -> List[Tuple[List[Tuple[int, int]], List[List[int]]]]:
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
                coincident_adc_values = [adc_data_list[j[0]] for j in current_group[0:2]]
                coincidences.append((current_group[0:2], coincident_adc_values))
            current_group = [(timestamps[i][0], timestamps[i][1])]    
            tstp_group = timestamps[i][1]                  
        else:
            if len(current_group) > 1:          
                coincident_adc_values = [adc_data_list[j[0]] for j in current_group[0:2]]
                coincidences.append((current_group[0:2], coincident_adc_values))
            current_group = [(timestamps[i][0], timestamps[i][1])]
            tstp_group = timestamps[i][1]

    if len(current_group) > 1:
        coincident_adc_values = [adc_data_list[j[0]] for j in current_group]
        coincidences.append((current_group, coincident_adc_values))

    return coincidences

def _send_coincidences_to_queue(coincidences: List[Tuple[List[Tuple[int, int]], List[List[int]]]], queue: Queue):
    """
    Sends coincidences to the writer process via the queue.

    Args:
        coincidences (List[Tuple[List[Tuple[int, int]], List[List[int]]]]): List of coincidences.
        queue (Queue): Queue to send coincidences to the writer process.
    """
    for group, adc_values in coincidences:                
        queue.put((group, adc_values))        

def process_segment(file_path: str, order: list, start: int, end: int, threshold: int, queue: Queue, event_size: int = 56):
    """
    Processes a segment of the file in smaller chunks, detects coincidences,
    and sends results to the writer process.

    Args:
        file_path (str): Path to the file.
        start (int): Start byte position for this segment.
        end (int): End byte position for this segment.
        threshold (int): Threshold for coincidence detection.
        queue (Queue): Queue to send coincidences to the writer process.
        event_size (int): Size of each event in bytes.
    """
    buffer_size = 10000 * event_size  # Define a buffer size for smaller chunks
    cnt_chunks = 1

    with open(file_path, "rb") as file:
        current_position = start
        with tqdm(total=end - start, unit="B", unit_scale=True, unit_divisor=1024, desc=f"Process {start // (end - start)}") as pbar:
            while current_position < end:
                # Read a smaller chunk from the file
                read_size = min(buffer_size, end - current_position)
                buffer = file.read(read_size)
                current_position += read_size
                pbar.update(read_size)

                # Extract timestamps and ADC data
                timestamps, adc_data_list = _extract_data(buffer, event_size, order)

                # Detect coincidences
                coincidences = _detect_coincidences(timestamps, adc_data_list, threshold)

                # Send coincidences to the writer process
                _send_coincidences_to_queue(coincidences, queue)
                
                # Break at certain number of events
                # if cnt_chunks == 20:
                #     break
                cnt_chunks += 1

            pbar.close()



if __name__ == "__main__":
    # Example usage
    # Read the file path from command-line arguments
    file_path = "P:/Valencia/I3M/Proyectos/DeepBrain/data/05022024_FOV0_0_600s.raw"
    output_file = "P:/Valencia/I3M/Proyectos/DeepBrain/data/05022024_FOV0_0_600s_coincidences.txt"
    # order = [1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    num_processes = 6
    threshold = 10000
    file_size = os.path.getsize(file_path)
    chunk_size = file_size // num_processes

    # Create a Queue for communication
    queue = Queue()

    # Start writer process
    writer = Process(target=writer_process, args=(output_file, queue))
    writer.start()

    # Start worker processes
    processes = []
    for i in range(num_processes):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_processes - 1 else file_size
        process = Process(target=process_segment, args=(file_path, order, start, end, threshold, queue))
        processes.append(process)
        process.start()

    # Wait for workers to finish
    for process in processes:
        process.join()

    # Signal the writer to stop
    queue.put("DONE")
    writer.join()
