import sys

from src.adc_reader import read_raw_file


def main():
    # Example usage
    # Read the file path from command-line arguments
    file_path = sys.argv[1]
    # order = [1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    # order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    order = range(33)
    for event in read_raw_file(file_path):
        ctrl = event["CTRL"]
        digital_signals = event["D3-D0"]
        timestamp = event["Timestamp"]
        s = event["S"]
        module_id = event["ModuleID"]
        adc_channels = event["ADC Channels"]
        print(f"CTRL: {ctrl}")
        print(f"D3-D0: {digital_signals}")
        print(f"Timestamp: {timestamp}")
        print(f"S: {s}")
        print(f"ModuleID: {module_id}")
        print("ADC Channels: ", end="")
        for i in order:
            print(f"{adc_channels[i]} ", end="")
        print()
        print(f"Length of ADC Channels: {len(adc_channels)}")
        input("Press Enter to continue...")


if __name__ == "__main__":
    main()
