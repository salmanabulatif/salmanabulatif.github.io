---
date:
  created: 2026-05-01
  updated: 2026-10-24
readtime: 20
authors:
    - salmanabulatif
---

# From Unboxing to Zero Day: Satellite Receiver Security Research

When I first picked up a GX6607 based satellite receiver I had one question in mind. How secure is this thing? What followed was months of hardware hacking, firmware analysis, reverse engineering and live testing across several receivers. This post is the updated technical companion to my talks at XCon and SANS CyberThreat 2026 in London, including what I got wrong the first time around.

<!-- more -->

## What Changed Since the First Version of This Post

Research is iterative, and some of my early conclusions did not survive a second look. In the spirit of separating what is confirmed from what is inferred, here is what I corrected:

- **There were two firmware platforms, not one.** The UART boot log and the CH341A flash dump came from an older eCos based receiver with a 4 MB SPI NOR chip. The GX6607 receiver, where I found the root shell, runs Linux from a 128 MB SPI NAND flash. Both appear in this post, and each section says which device it is about.
- **"JBOOT" and "Blowfish" were binwalk false positives.** binwalk reported JBOOT headers (one claiming a 1.1 GB image inside a 10 MB file), ESP32 images and Blowfish encrypted regions. None of them were real. The boot log clearly names the bootloader as zeroBoot.
- **The boot log did not give me dvbapp's load address.** dvbapp is an ELF file and carries its own base address. The addresses in the boot log belong to the eCos firmware.
- **The buffer overflow is confirmed in code, not exploited.** I have not demonstrated a path that delivers attacker-controlled data to it, so I no longer assign it an attack vector.

## The Targets

The main target is a satellite set-top box built on the Nationalchip GX6607 SoC. It is sold at retail as the GAZAL BOSS S5 FOREVER, while its firmware identifies itself as GAZAL BOSS 03 FOREVER. That mismatch matters: it shows the same firmware is shared across models and brands. The box runs embedded Linux, and a single 10 MB ARM application called dvbapp handles almost everything: DVB reception, conditional access, IPTV, OTT apps, the user interface and network services.

I also tested older receivers from the same market, built in different years. One of them runs eCos, a real-time operating system, from a 4 MB SPI NOR chip. That box is where the UART and flash dumping work below took place.

## Opening the Device and Finding UART (eCos receiver)

The first step in any hardware engagement is getting eyes on the PCB. After removing the case I photographed the board and identified the SoC, the SPI flash in an SOIC8 package, the DDR memory and a set of test pads near the edge of the board.

I used a multimeter to profile each pad against ground while the device was powered. The TX line fluctuates during boot and then settles at 3.3 V. The RX line sits at 3.3 V waiting for input. Ground is ground.

The baud rate took longer. My first logic analyzer capture at 20 kHz showed what looked like a clean clock signal. The problem was my setup, not the device: 20 kHz is far too slow to resolve any standard baud rate above 2400. After recapturing at 4 MHz the pulses had the variable widths of real UART framing, and I calculated the baud rate from the shortest pulse in PulseView.

The second lesson was about terminal software. Minicom is forgiving about baud rate mismatches and can show plausible text at the wrong rate. Picocom is strict. If minicom looks clean but picocom shows garbage, your baud rate is wrong. Verify with the strict tool before trusting what you see.

### What the Boot Log Gave Away

Power cycling the board produced a complete boot log. In about ten seconds the device told me:

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
file:system_others.c,line176: LZMA Unzip module m_info.lma:=0x1b924,m_info.lma_size:=0x83a41
file:system_others.c,line213: bootloader m_info.vma:=0x80010000
SQMACSQMAC
XCAS_V21.09.01
ISH_V21.09.01
```

- **The boot chain:** a zeroBoot first stage, a second stage from SPI flash, then eCos.
- **Factory debug output left on:** pages of DDR calibration registers, source file names and line numbers, printed on every boot of a consumer product.
- **No visible integrity check:** nothing in the boot chain reports a signature or integrity verification before handing over to the next stage.
- **Old code:** the loader version string is dated 26 December 2011.
- **Instructions for reverse engineering:** the log states where the compressed application sits in flash, how large it is, and the address it is loaded to. That is exactly what you need to extract it and load it into Ghidra correctly.
- **Card sharing modules load at boot:** the conditional access and softcam modules are part of the boot sequence, not an add-on.

## Dumping the Firmware (eCos receiver)

The CH341A programmer with an SOIC8 clip read the 4 MB GigaDevice SPI NOR chip without trouble once I followed three rules: the board must be fully powered off, the programmer must use 3.3 V logic, and a short, good quality USB cable avoids the brownouts that make the CH341A disconnect mid-read.

I always take three reads and compare their hashes. If they differ, the clip connection is unreliable and anything you analyse is meaningless.

The channel database was readable in plain text and listed transponders for Nilesat, Badr, Hotbird, Astra and dozens more, which reveals the market the device was built for. The dump also held cached runtime data, including Wi-Fi credentials from whoever configured the device before me. If you sell or repair one of these boxes without wiping it, your home network password goes with it.

### A Lesson About Trusting Tools

My first pass leaned heavily on binwalk's signature scan, and I reported what it said: JBOOT headers, Blowfish encrypted regions, ESP32 images. On review, these were false positives. Short magic-number signatures match random bytes in large binaries all the time. Treat binwalk's output as a list of leads to verify, never as findings.

## Getting Inside the GX6607 (Linux receiver)

The GX6607 box took a very different path. A network scan showed telnet open on port 23.

### The Critical Finding: Root Without a Password

Connecting to port 23 gives a root shell immediately. There is no login prompt. The cause is two lines in the read-only system image:

- The init script starts `telnetd` with a custom login program, `/etc/autologin`.
- That script simply runs `login -f root`, which logs in as root without checking a password.

On top of that, the root account in `/etc/passwd` has no password at all.

Because this lives in the read-only squashfs image, it is not a leftover setting a user could change. It was built into the firmware.

From this shell an attacker can read every credential stored on the device, harvest its configuration, watch its traffic, and use it as a foothold into the rest of the home network.

- **CWE:** CWE-306 (Missing Authentication for Critical Function), CWE-1188 (Insecure Default Initialization), CWE-912 (Hidden Functionality)
- **CVSS 3.1:** 9.6 Critical (`AV:A/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H`) for a typical home network. If the device is directly reachable from the internet, for example through a cellular connection with a public IP, the attack vector becomes network and the score 10.0.

### A Second Unauthenticated Service

dvbapp also runs an HTTP service on a local network port. That service returns files from the device's filesystem to anyone who asks, with no authentication. It is independent of telnet, so it remains exposed even if telnet were disabled. I have included it in my disclosure.

### What Is Listening

dvbapp alone listens on six TCP ports. telnetd listens on every interface, IPv4 and IPv6. Every way the box connects to a network (Ethernet, Wi-Fi or cellular) leads to the same root shell.

## Static Analysis in Ghidra (dvbapp)

dvbapp is a 32-bit little-endian ARM ELF. Ghidra picks up its base address from the ELF headers, which matched the process memory map on the running device. With the Function Start Search and Aggressive Instruction Finder analysers enabled, Ghidra recovered most of the function structure without symbols.

My approach is strings first. A distinctive string is an anchor: find it, follow its cross-references to the function that uses it, and work outward.

### Hardcoded Authentication Secrets

The IPTV service dispatcher builds an authentication token for its backend API by hashing a secret salt together with the device's MAC address, serial number and model name. The salt is hardcoded in the binary. A second hardcoded key is used by the request builder for another IPTV service.

Anyone who extracts these secrets can generate valid tokens for any device whose identifiers they know, without a subscription. I have deliberately left the values and the exact token format out of this post.

- **CWE:** CWE-798 (Use of Hard-coded Credentials), CWE-321 (Use of Hard-coded Cryptographic Key)
- **CVSS 3.1:** 7.5 High

### A Stack Buffer Overflow, Confirmed in Code Only

One small function copies a caller-supplied number of bytes into a fixed 260-byte stack buffer with no bounds check, then passes the buffer to a hardware write routine. The code around its callers checks an XOR checksum and a status word typical of smartcard communication, so it appears to be part of the conditional access path.

This is a real coding flaw (CWE-121). However, the callers I traced pass small fixed or byte-bounded lengths, and I have not shown a way for outside data to reach it with an oversized length. I therefore report it as a code-quality finding without an attack vector or exploitability claim.

### No Memory Protections

The process memory map shows every region of dvbapp, including the stack and heap, as readable, writable and executable. There is no ASLR, no stack canaries and no NX. These protections are standard on modern Linux, and their absence turns any memory corruption bug into a far more serious problem.

### A Correction: The Broadcast Parser Bug That Was Not

Earlier I suspected an unchecked index in what I took for a DVB broadcast table parser. A closer look showed the structure holds IPTV stream records, not broadcast tables, and the index is range-checked before use. I have dropped that finding.

## What They Left Behind

Live inspection of the GX6607 turned up a long trail of development and configuration leftovers:

- **Developer artefacts:** build paths naming individual developers, a developer's Wi-Fi network credentials, and a debug script that mounts a development server over NFS.
- **A hardcoded Wi-Fi hotspot configuration** with a weak, guessable default passphrase.
- **A preloaded cellular database:** 101 carrier profiles for USB 3G/4G modems across more than 20 countries, including Jordan's operators. These are public APN defaults, not leaked secrets, but they show cellular support is built in.
- **Chinese DNS by design:** the cellular dialer scripts, credited to two named developers, deliberately keep a resolver configuration pointing at China Telecom (CHINANET) DNS servers after a cellular link comes up. A comment in the script says so explicitly. On that path, every domain the box looks up goes to those servers.
- **A default China Telecom CDMA login** left over from the reference design.
- **A boot video from a writable partition.** At every boot, a root process opens a video file from a writable flash partition, and a companion file suggests that video can be downloaded from a remote URL. Whoever controls that content controls what a media parser running as root opens at boot. This is an attack surface I have not yet tested.
- **A leftover OpenVPN DNS file,** but no VPN client and no code in dvbapp that uses one. I mention it only so nobody overreads it.

## Assessing the RF Attack Surface

Assessing RF does not start with a transmitter. It starts with knowing which radios the box has, how each one works, and where each one leads.

| Radio | How it works on this box | Status |
|---|---|---|
| Satellite (DVB-S/S2) | Tuner → hardware demux → descrambler. The kernel also ships IP-over-satellite encapsulation paths (GSE, ULE). | Observed |
| Wi-Fi | USB Wi-Fi/Bluetooth combo chip in client mode, plus a shipped hotspot configuration | Configuration confirmed |
| Wi-Fi Direct | A `p2p0` interface and a screen-mirroring (Wi-Fi Display) module | Inferred from names |
| Bluetooth LE | Used by the remote control | Observed |
| Cellular | USB 3G/4G modem support with PPP dialer scripts | Configuration confirmed |

Three conclusions follow:

1. **The broadcast signal is not authenticated.** Anyone who can transmit on the right frequency is talking to the box's parsers.
2. **The decryption keys come from the internet, not from a smartcard** (see the next section).
3. **Every radio leads to the same root shell,** because telnet listens on every interface.

## IKS: Card Sharing Built Into the Firmware

Legitimate pay-TV works like this. Each channel is scrambled with a key that changes every few seconds. That key is broadcast in encrypted form, and only the subscriber's smartcard can decrypt it.

IKS (Internet Key Sharing) replaces the smartcard with a server on the internet. The receiver sends the encrypted key message to a server that holds a genuine subscription card and gets the decrypted key back. The video itself still comes from the dish; only a few bytes cross the internet.

On the GX6607 I confirmed:

- **Live, persistent connections** to card-sharing servers. One is hosted in the Netherlands on infrastructure registered through a Belize entity.
- **The card-sharing protocols** (CCcam, Newcamd, MGcamd and OSCam related code) built into dvbapp.
- **Device identification at activation:** the activation request sends the device's MAC address and serial number to the operator.
- **No smartcard in use:** decryption keys come only from the internet.

Why this matters for security:

- The box keeps a permanent, encrypted, outbound connection to anonymous operators running illegal infrastructure.
- dvbapp, running as root with no memory protections, parses everything those servers send back.
- The operators know exactly which device is which.

Card sharing is illegal in many countries, including the UK. I document the infrastructure as evidence and do not use or demonstrate the service.

## From Vulnerable Box to Residential Proxy

In July 2026 the FBI and IRS-CI seized the domains of NetNut, a residential proxy service run by the Nasdaq-listed Israeli firm Alarum, together with Google, Lumen and Shadowserver. Researchers had linked NetNut to "Popa", a botnet of at least two million smart TVs and TV streaming boxes. These devices either shipped with proxy software or picked it up through apps, and customers rented their home internet connections. In a single week Google observed 316 distinct threat actor groups using NetNut exit nodes, including criminal and espionage groups. Alarum rejects the botnet label and says its software is consent-based.

I did not find any proxy software on the GX6607, and I am not claiming it is part of NetNut. But it fits the profile almost perfectly:

- always on, behind a residential IP;
- root with no password, on every interface;
- already in contact with unknown servers by design;
- nobody monitors it and nobody patches it.

It is not a botnet today. It is a ready-made node for one.

## How These Devices Get Into Homes

- **Diaspora demand.** Communities abroad want the channels they grew up with, so these receivers travel with them to Europe and North America. The users are the victims, not the threat.
- **Local label, foreign build.** The boxes I bought say "Made in Jordan". Inside are a Chinese chipset, Chinese developer build paths and Chinese DNS servers. I expect the same pattern in other markets.
- **Borders check paperwork, not firmware.** Customs checks duties and origin, and radio regulators check frequencies. Nobody inspects the code.
- **Nobody's job.** It is not a computer, so IT does not own it. It is not a phone, so telecom security does not look at it.

## Responsible Disclosure

I attempted to notify the vendors and distributors before publishing and have not received a response. A CVE request has been filed for the unauthenticated root shell. The findings were presented at XCon and will be presented at SANS CyberThreat 2026 in London on 17 November 2026. Secrets, credentials and step-by-step exploitation details are withheld from all public material.

## Lessons Learned

- **Sample rate matters.** Too low a sample rate does not give you garbage. It gives you plausible output that is completely wrong. Confirm with a strict tool.
- **Know which device a result came from.** I mixed evidence from two platforms early on, and it led me to wrong conclusions. Label every artefact with its source.
- **Verify every tool's output.** binwalk signatures, decompiler output and string matches are leads, not findings.
- **The boot log is a gift,** when it belongs to the firmware you are analysing.
- **Separate confirmed, observed and inferred.** It makes every claim you keep stronger.
- **The worst bug was the simplest.** The most severe finding needed no firmware knowledge, no UART and no flash dump. It needed one telnet command.

## Tools Used

Hardware: a CH341A programmer with an SOIC8 clip, a 3.3 V USB-to-UART adapter, a logic analyzer with PulseView, a multimeter, and an RTL-SDR for passive spectrum observation.

Software: flashrom, binwalk, Ghidra, picocom and minicom, standard Linux tools on the device itself, and public WHOIS, DNS and RIPE data for infrastructure research.

## Sources

- [FBI Seizes NetNut Proxy Platform, Popa Botnet (KrebsOnSecurity)](https://krebsonsecurity.com/2026/07/fbi-seizes-netnut-proxy-platform-popa-botnet/)
- ['Popa' Botnet Linked to Publicly-Traded Israeli Firm (KrebsOnSecurity)](https://krebsonsecurity.com/2026/06/popa-botnet-linked-to-publicly-traded-israeli-firm/)
- [Google's Continued Disruption of Malicious Residential Proxy Networks (Google Threat Intelligence Group)](https://cloud.google.com/blog/topics/threat-intelligence/google-continued-disruption-residential-proxy-networks)
