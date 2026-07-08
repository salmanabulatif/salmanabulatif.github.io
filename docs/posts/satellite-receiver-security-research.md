---
date:
  created: 2026-05-01
readtime: 15
authors:
    - salmanabulatif
---

# From Unboxing to Zero Day: Satellite Receiver Security Research

When I first picked up a GX6607 based satellite receiver I had one question in mind. How secure is this thing? What followed was several months of hardware hacking, firmware analysis, and

<!-- more -->

## The Target

The device in question is a GX6607 based satellite set-top box running an embedded Linux system with a 10MB application binary called dvbapp at the heart of everything. The GX6607 is a Nationalchip SoC and it powers a wide range of satellite receivers sold under different brand names across the region. The firmware I analyzed belongs to the Gazal Boss 03 Forever but the same codebase and the same vulnerabilities exist across multiple devices built on this platform.

Before touching any hardware I spent time understanding what this chip family actually does. It handles DVB signal reception, conditional access system decryption, IPTV streaming, OTT service integration, and UI rendering all in one binary. That is a significant attack surface for a device that most people never think about from a security perspective.

## Opening the Device and Finding UART

The first step in any hardware security engagement is getting eyes on the PCB. After removing the case I photographed the board from multiple angles and began identifying components. The main SoC was clearly labeled and surrounding it were the usual suspects: SPI NOR flash in an SOIC8 package, DDR memory, and a handful of test pads near the edge of the board.

Those test pads are always interesting. On embedded Linux devices they almost always carry a UART debug console and this board was no exception. I used a multimeter to profile each pad against ground while the device was powered on. The TX line fluctuates during boot and then settles to a steady 3.3V when idle. The RX line sits at 3.3V waiting for input. Ground is ground. This process took about ten minutes.

The trickier problem was the baud rate. I connected my logic analyzer and captured the TX output during boot. The first capture taught me something important about sample rates. I had the analyzer set to 20 kHz which is completely inadequate for detecting any standard baud rate above 2400. The signal I captured looked like a perfectly regular clock rather than UART data which was the first clue that something was wrong with my setup rather than the device.

After recapturing at 4 MHz the picture became clear. The signal had variable pulse widths consistent with actual UART framing. I measured the shortest pulse width manually in PulseView and calculated the baud rate from there. The device was running at a non-standard rate that was only detectable with a high enough sample rate.

The other lesson from this process was the difference between minicom and picocom. Minicom is quite forgiving about baud rate matching and will display something that looks like readable output even at the wrong rate. Picocom is strict. If you see clean output in minicom but garbage or nothing in picocom that is a strong signal that your baud rate is wrong and minicom is tolerating errors rather than correctly decoding the data. Always verify with picocom before trusting what you see.

Once connected at the correct baud rate and power cycling the device I was rewarded with a complete boot log. This is where hardware UART access earns its keep. In about ten seconds the device tells you its entire life story.

```
zeroBoot v1.0 for 1506F
SPI SCS flow
jump to second boot!!
Boot form SPI v1.11
DRAM-512M
copy fw to RAM
1ST BOOT
LD eCos v1.16 1506f
20111226-V1.1, RAM_START:=0x81600000
bootloader m_info.vma:=0x80010000
XCAS_V21.09.01
ISH_V21.09.01
DQCAM_V21.09.02
```

From this single boot log I now knew the bootloader chain, the RTOS, the RAM base address for Ghidra, the conditional access system version, and the load address. That last piece of information is critical for static analysis because it tells you exactly where to base the firmware image in your disassembler.

## Dumping the Firmware

With UART confirmed I turned to the flash chip. The SOIC8 package on this board is a standard SPI NOR chip and the CH341A programmer handles these perfectly with an SOIC8 test clip. The key requirements are that the target board is completely powered off before connecting the clip, that you are using a 3.3V logic level rather than 5V, and that you use a short quality USB cable to avoid the brownout issues that cause the CH341A to disconnect mid-transfer.

I always do three separate reads and compare their MD5 hashes before trusting a dump. If the hashes do not match the clip connection is unreliable and any analysis you do on that data is meaningless.

```bash
flashrom -p ch341a_spi -r dump1.bin
flashrom -p ch341a_spi -r dump2.bin
flashrom -p ch341a_spi -r dump3.bin
md5sum dump1.bin dump2.bin dump3.bin
```

Once I had a verified dump I ran it through binwalk for an initial survey. The entropy analysis was the most informative first step. High entropy regions near 1.0 indicate encryption or compression. The firmware had clear sections of each. What stood out immediately were regions identified as Blowfish-448 CBC encrypted data alongside JBOOT headers and zlib compressed modules. This told me the device uses the JBOOT bootloader format and that the conditional access system data is encrypted with Blowfish.

The channel database sections were completely readable in plaintext and contained satellite transponder information for Nilesat, Badr, Hotbird, Astra, and dozens of others. The channel list itself contained service names that gave away the regional market this device was built for.

## What the Firmware Revealed

Running strings against the dump before any deep analysis surfaces a lot of useful information quickly. In this firmware I found a TFTP upgrade path hardcoded in the binary pointing to an internal development server. I found what appeared to be a test encryption key labeled TEST KEY in plaintext. And I found a runtime blob that had been cached on the device which contained WiFi credentials belonging to whoever had configured the device before it reached me.

That last finding is worth dwelling on. The device caches connection information in flash and that cache is fully readable by anyone with a CH341A and ten minutes. If you sell your satellite receiver or send it for repair without clearing this data your home network credentials go with it.

The blob also contained the JSON configuration for the devices OTT streaming services including a list of supported applications like YouTube, Al Jazeera, Dailymotion, and TikTok. More interestingly it contained what appeared to be an AES key field set to a placeholder value suggesting the production key management was not functioning as intended in this firmware build.

## Static Analysis in Ghidra

Loading the dvbapp ELF binary into Ghidra was straightforward once I had the correct architecture from binwalk. The binary is ARM 32-bit little-endian and the load address from the boot log confirmed exactly where to base it. With Function Start Search and Aggressive Instruction Finder enabled Ghidra recovered a substantial portion of the function structure even without symbols.

The approach I use for an unfamiliar binary is to start with strings. Interesting strings are anchors. You find the string in the binary, follow its cross-references to the function that uses it, and from there you can understand what that function does and trace outward. The string CA_Write led me directly to the conditional access write path. A hardcoded authentication salt string led me to the IPTV authentication implementation. The string rom_net_upgrade.bin pointed at the firmware update handler.

### The IPTV Authentication Scheme

One function I spent significant time on was the service dispatcher for the DragonPlus IPTV backend. This function constructs authenticated API requests and the authentication scheme it uses is worth documenting.

The device builds an authentication token by concatenating a hardcoded salt value with the device MAC address, serial number, and model identifier, then computing MD5 over the result and hex-encoding it. This token goes into the key parameter of every API request sent to the backend.

The problem is that the salt is hardcoded in the binary. Anyone who extracts it can forge authenticated requests to the backend API on behalf of any device. The salt combined with device identifiers that are often observable from the network or printed on the device label means an attacker can generate valid authentication tokens without owning a legitimate subscription. A simplified version of what the device does internally looks like this:

```python
import hashlib

salt  = "<hardcoded_salt>"
mac   = "AA:BB:CC:DD:EE:FF"
sn    = "123456789"
model = "GX6607"

token = hashlib.md5(f"{salt},{mac},{sn},{model}".encode()).hexdigest()
url   = f"http://backend/api.php?action=live&mac={mac}&sn={sn}&key={token}"
```

This is classified as CWE-321 (Use of Hard-coded Cryptographic Key) and scores 7.5 High on CVSS 3.1 with a network attack vector.

### The Buffer Overflow in the Hardware Write Path

The most pedagogically interesting finding from a code analysis perspective is a stack buffer overflow in the hardware interface write function. The entire function is nine lines long and the vulnerability is immediately visible.

```c
uint FUN_000baa00(int param_1, uint *param_2, uint param_3, uint param_4)
{
  uint uVar1;
  uint auStack_110 [65];    // fixed buffer: 65 x 4 = 260 bytes

  thunk_FUN_007b0530(auStack_110, param_2, param_3, param_4);
  // copies param_3 bytes from param_2 into a 260 byte buffer
  // param_3 has no upper bound check anywhere in the call chain

  uVar1 = FUN_00333640(param_1, (byte *)auStack_110, param_3);
  return uVar1 & ~((int)uVar1 >> 0x1f);
}
```

A fixed 260 byte stack buffer receives data via memmove. The size argument comes from the caller as an external parameter. There is no upper bound check anywhere in the function or in any of its callers. The source data and the copy length both originate from protocol parsing code that processes incoming DVB packet data.

When the copy length exceeds 260 bytes the write continues past the end of the buffer into the saved link register on the stack. On ARM this is the return address. Overwriting it with attacker-controlled data redirects execution when the function returns. Because this binary has no stack canaries, no ASLR, no NX, and no PIE there are no mitigations to bypass. The saved register is at a fixed predictable offset and the stack is executable.

The same function then passes the potentially corrupted buffer to the hardware transmission function. If the copy length is large this transmission reads past the end of the stack buffer and sends saved register values over the hardware channel creating an information leak alongside the overflow.

The vulnerability is CWE-121 with a CVSS score of 7.5 High reflecting an adjacent network attack vector since the trigger is a malformed protocol packet received over the network or IPTV stream rather than direct user input.

## The Critical Finding: Unauthenticated Root Shell

Everything above is interesting research. What I found next is a critical vulnerability.

Port 23 on the device accepts telnet connections. There is no authentication prompt. The connection immediately drops you into a root shell.

```
telnet 192.168.1.56
Connected to 192.168.1.56.
[root@GX6607:~]#
```

That is the entire attack. One command. No credentials. Full root access to a device that sits on home networks, handles conditional access decryption, stores WiFi passwords in flash, and runs a 10MB application binary with multiple additional vulnerabilities.

From this shell an attacker can read all credentials stored on the device, modify the firmware, install persistent backdoors that survive reboots, pivot to other devices on the home network, and extract conditional access keys from memory while they are actively in use.

This vulnerability is classified as CWE-306 (Missing Authentication for Critical Function) and CWE-1188 (Insecure Default Initialization). In a typical home deployment behind NAT it scores 9.6 Critical on CVSS 3.1. In deployments where the device is directly internet-exposed, which is not uncommon in certain ISP configurations across the region, it scores 10.0.

## Responsible Disclosure

I attempted to notify the vendors and distributors responsible for these devices before publishing this research. At the time of writing I have not received any response. The findings were presented publicly at XCon Security Conference in Beijing following the standard 90 day disclosure window. A CVE submission has been filed and is pending assignment.

## Lessons Learned

A few things from this research that I think are worth carrying forward.

Sample rate matters more than most people realize. Capturing UART at too low a sample rate does not give you garbage output. It gives you output that looks plausible but is completely wrong. Always verify with a tool that does strict baud rate matching before trusting what you see.

The boot log is more valuable than most researchers give it credit for. Five seconds of serial output told me the architecture, the load address, the RTOS, the bootloader chain, and the CAS version. That is weeks of work avoided.

Entropy analysis is a habit worth building. Looking at the shape of data before trying to parse it saves enormous amounts of time. If a region is at maximum entropy you need to understand the encryption before you can read it. If it is low entropy you can probably just start reading strings.

The most severe vulnerability on this device required no firmware knowledge, no UART access, and no flash extraction. It required one telnet command. Hardware security research is not just about finding sophisticated memory corruption bugs. Sometimes the most impactful finding is the most obvious one.

## Tools Used

The hardware side of this engagement used a CH341A USB programmer with an SOIC8 test clip, a USB to UART adapter at 3.3V logic levels, and a logic analyzer with PulseView for baud rate identification.

The software side relied on flashrom for firmware extraction, binwalk for initial analysis and recursive extraction, Ghidra for static analysis of the ELF binary, picocom and minicom for UART interaction, and standard Linux command line tools throughout.
