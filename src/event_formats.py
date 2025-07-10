import numpy as np

EVENT_DTYPE_V1 = np.dtype(
    [
        ("CTRL", "u1"),
        ("Timestamp", ">u4"),
        ("S_ModuleID", "u1"),
        ("ADC_Data", "50B"),
    ]
)

EVENT_DTYPE_V2 = np.dtype(
    [
        ("Header0", "u1"),
        ("Timestamp_1", "u1"),
        ("Timestamp_2", "u1"),
        ("Timestamp_3", "u1"),
        ("Timestamp_4", "u1"),
        ("Header1", "u1"),
        ("ADC_Data", "50B"),
    ]
)
