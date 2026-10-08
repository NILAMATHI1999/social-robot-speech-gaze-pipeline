#!/usr/bin/env python3

import struct
import time
import usb.core


device = usb.core.find(
    idVendor=0x2886,
    idProduct=0x0018
)

if device is None:
    raise RuntimeError("ReSpeaker not found")


def read_value(group, offset):
    command = 0x80 | offset | 0x40

    response = device.ctrl_transfer(
        0xC0,
        0,
        command,
        group,
        8,
        100000,
    )

    return struct.unpack(
        "ii",
        bytes(response),
    )[0]


print(
    "Speak from different sides. "
    "Press Ctrl+C to stop."
)

try:
    while True:
        vad = read_value(19, 32)
        direction = read_value(21, 0)

        print(
            f"Voice: {'YES' if vad else 'NO'} | "
            f"Direction: {direction} degrees"
        )

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nStopped.")
