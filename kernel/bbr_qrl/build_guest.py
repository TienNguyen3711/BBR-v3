"""Assemble a minimal initramfs from public build-image packages only."""
from pathlib import Path
import os,re,shutil,subprocess
root=Path('/lab/rootfs');root.mkdir(exist_ok=True)
for d in ['bin','sbin','usr/bin','usr/sbin','proc','sys','dev','tmp','run','etc','lib','modules']:
    (root/d).mkdir(parents=True,exist_ok=True)
shutil.copy2('/bin/busybox',root/'bin/busybox')
for name in ['sh','mount','ip','insmod','rmmod','poweroff','sleep','cat','echo']:
    os.symlink('busybox',root/'bin'/name)
shutil.copytree('/usr/lib/python3.12',root/'usr/lib/python3.12',dirs_exist_ok=True)
files=['/usr/bin/python3.12','/usr/sbin/tc']
files += [str(p) for p in Path('/usr/lib/python3.12/lib-dynload').glob('*.so')]
for f in files:
    target=root/f.lstrip('/');target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists(): shutil.copy2(f,target)
    dependencies=subprocess.run(['ldd',f],capture_output=True,text=True).stdout
    for dep in re.findall(r'(/[^\s()]+)',dependencies):
        dest=root/dep.lstrip('/');dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): shutil.copy2(dep,dest)
os.symlink('python3.12',root/'usr/bin/python3')
for name in ['tcp_bbr.ko','bbr_qrl.ko']:
    shutil.copy2('/lab/artifacts/'+name,root/'modules'/name)
shutil.copy2('/lab/guest_audit.py',root/'guest_audit.py')
(root/'init').write_text('''#!/bin/sh
export PATH=/bin:/usr/bin:/usr/sbin
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev
mount -t debugfs debugfs /sys/kernel/debug
ip link set lo mtu 1500
ip link set lo up
insmod /modules/tcp_bbr.ko || poweroff -f
insmod /modules/bbr_qrl.ko || poweroff -f
/usr/sbin/tc qdisc add dev lo root netem delay 10ms rate 20mbit limit 1000 || poweroff -f
/usr/bin/python3 /guest_audit.py
rmmod bbr_qrl
poweroff -f
''')
(root/'init').chmod(0o755)
subprocess.run('find . -print0 | cpio --null -o --format=newc | gzip -1 > /lab/artifacts/initramfs.gz',cwd=root,shell=True,check=True)

import hashlib
artifacts=Path('/lab/artifacts')
(artifacts/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in sorted(artifacts.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
