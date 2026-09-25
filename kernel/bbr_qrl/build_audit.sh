#!/bin/sh
set -eu
cd /lab/linux
make tinyconfig
scripts/config --enable 64BIT --enable ARM64 --enable MMU --enable ARCH_VIRT \
 --enable SMP --enable HIGH_RES_TIMERS --enable PRINTK --enable TTY --enable SERIAL_AMBA_PL011 --enable SERIAL_AMBA_PL011_CONSOLE \
 --enable BINFMT_ELF --enable BINFMT_SCRIPT --enable BLOCK --enable BLK_DEV_INITRD --enable RD_GZIP \
 --enable DEVTMPFS --enable DEVTMPFS_MOUNT --enable PROC_FS --enable SYSFS --enable TMPFS --enable DEBUG_FS \
 --enable MODULES --enable MODULE_UNLOAD --enable NET --enable INET --enable UNIX --enable NETDEVICES \
 --enable NET_NS --enable NAMESPACES --enable VETH --enable DUMMY --enable NET_SCHED --enable NET_SCH_NETEM \
 --enable NET_SCH_FQ --enable NET_SCH_TBF --enable INET_DIAG --enable INET_TCP_DIAG \
 --enable TCP_CONG_ADVANCED --module TCP_CONG_BBR --enable TCP_CONG_CUBIC \
 --enable FUTEX --enable EPOLL --enable EVENTFD --enable TIMERFD --enable POSIX_TIMERS \
 --enable FHANDLE --enable MULTIUSER --enable AIO --enable NETLINK_DIAG \
 --disable DEBUG_INFO --disable DEBUG_INFO_BTF --disable WERROR
make olddefconfig
make -j4 Image modules
python3 /lab/adapter/prepare_source.py net/ipv4/tcp_bbr.c --out /lab/adapter/tcp_bbr_qrl.c
cp net/ipv4/tcp_dctcp.h /lab/adapter/
make -C /lab/adapter KDIR=/lab/linux -j4
mkdir -p /lab/artifacts
cp arch/arm64/boot/Image .config /lab/artifacts/
cp net/ipv4/tcp_bbr.ko /lab/adapter/bbr_qrl.ko /lab/artifacts/
sha256sum /lab/artifacts/* > /lab/artifacts/SHA256SUMS
