// SPDX-License-Identifier: GPL-2.0-only
/* Linked into bbr_qrl.ko. Single-flow lab only. Commands latch on eligible
 * CRUISE entry. Clear/one-second lease expiry restore native gain on ACK,
 * including BBR's fast path. Multiple active sockets disable overrides.
 */
#include <linux/module.h>
#include <linux/debugfs.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/spinlock.h>
#include <linux/jiffies.h>
#include "gain_parser.h"
#include "bbr_qrl_ctl.h"

static DEFINE_SPINLOCK(ctl_lock);
static struct dentry *ctl_dir;
static struct sock *owner;
static unsigned int active;
static int requested = -1, latched = -1, was_cruise;
static int current_phase = -1, applied = 256, native = 256, cruise;
static unsigned long refreshed;
static u64 cruise_entries;
static u64 phase_acks[8], override_acks, rejected_multiflow_acks, canceled_acks;
static u64 guarded_cruise_acks;

void bbr_qrl_attach(struct sock *sk)
{
	spin_lock_bh(&ctl_lock);
	if (!active) {
		owner = sk;
		was_cruise = 0;
		latched = -1;
	}
	active++;
	if (active > 1) latched = -1;
	spin_unlock_bh(&ctl_lock);
}

void bbr_qrl_detach(struct sock *sk)
{
	spin_lock_bh(&ctl_lock);
	if (active) active--;
	if (owner == sk) {
		owner = NULL;
		requested = latched = -1;
		was_cruise = cruise = 0;
		current_phase = -1;
	}
	spin_unlock_bh(&ctl_lock);
}

int bbr_qrl_apply(struct sock *sk, int phase, int eligible, int native_gain)
{
	int result = native_gain;
	spin_lock_bh(&ctl_lock);
	if (phase >= 0 && phase < 8) phase_acks[phase]++;
	if (phase == 6 && !eligible) guarded_cruise_acks++;
	if (active != 1 || owner != sk) {
		rejected_multiflow_acks++;
		latched = -1;
		cruise = 0;
		if (owner == sk) was_cruise = phase == 6;
		goto out;
	}
	if (phase == 6 && !was_cruise) cruise_entries++;
	if (requested < 0 || time_after_eq(jiffies, refreshed + HZ)) {
		if (latched >= 0) canceled_acks++;
		latched = -1;
	} else if (eligible && !was_cruise) {
		latched = requested;
	}
	if (!eligible) latched = -1;
	was_cruise = phase == 6;
	cruise = !!eligible;
	if (eligible && latched >= 0) {
		result = latched;
		override_acks++;
	}
out:
	current_phase = phase;
	applied = result;
	native = native_gain;
	spin_unlock_bh(&ctl_lock);
	return result;
}

static ssize_t gain_write(struct file *file, const char __user *ubuf,
			  size_t count, loff_t *ppos)
{
	char buf[32];
	int gain, err;
	if (!count || count >= sizeof(buf)) return -EINVAL;
	if (copy_from_user(buf, ubuf, count)) return -EFAULT;
	err = bbr_qrl_parse_gain(buf, count, &gain);
	if (err) return err;
	spin_lock_bh(&ctl_lock);
	requested = gain;
	refreshed = jiffies;
	spin_unlock_bh(&ctl_lock);
	return count;
}

static ssize_t gain_read(struct file *file, char __user *ubuf, size_t count, loff_t *pos)
{
	char buf[32];
	int gain, len;
	spin_lock_bh(&ctl_lock);
	gain = time_after_eq(jiffies, refreshed + HZ) ? -1 : requested;
	spin_unlock_bh(&ctl_lock);
	/* Readback shows quantized gain, not the decimal command sent. */
	len = gain < 0 ? scnprintf(buf, sizeof(buf), "-1.000000\n") :
		scnprintf(buf, sizeof(buf), "%d.%06d\n", gain / 256, (gain % 256) * 1000000 / 256);
	return simple_read_from_buffer(ubuf, count, pos, buf, len);
}

static ssize_t cruise_read(struct file *file, char __user *ubuf, size_t count, loff_t *pos)
{
	char buf[4];
	int len = scnprintf(buf, sizeof(buf), "%d\n", READ_ONCE(cruise));
	return simple_read_from_buffer(ubuf, count, pos, buf, len);
}

static ssize_t status_read(struct file *file, char __user *ubuf, size_t count, loff_t *pos)
{
	char buf[768];
	int len;
	spin_lock_bh(&ctl_lock);
	len = scnprintf(buf, sizeof(buf),
		"active=%u phase=%d eligible=%d requested=%d latched=%d applied=%d native=%d "
		"entries=%llu override_acks=%llu canceled_acks=%llu multiflow_acks=%llu guarded_cruise_acks=%llu "
		"startup=%llu drain=%llu probe_rtt=%llu up=%llu down=%llu cruise=%llu refill=%llu\n",
		active, current_phase, cruise, requested, latched, applied, native,
		cruise_entries, override_acks, canceled_acks, rejected_multiflow_acks, guarded_cruise_acks,
		phase_acks[0], phase_acks[1], phase_acks[3], phase_acks[4], phase_acks[5], phase_acks[6], phase_acks[7]);
	spin_unlock_bh(&ctl_lock);
	return simple_read_from_buffer(ubuf, count, pos, buf, len);
}

static const struct file_operations gain_fops = {
	.owner = THIS_MODULE, .read = gain_read, .write = gain_write, .llseek = default_llseek,
};
static const struct file_operations cruise_fops = {
	.owner = THIS_MODULE, .read = cruise_read, .llseek = default_llseek,
};
static const struct file_operations status_fops = {
	.owner = THIS_MODULE, .read = status_read, .llseek = default_llseek,
};

int bbr_qrl_ctl_init(void)
{
	struct dentry *entry;
	ctl_dir = debugfs_create_dir("tcp_bbr_qrl", NULL);
	if (IS_ERR_OR_NULL(ctl_dir)) return ctl_dir ? PTR_ERR(ctl_dir) : -ENOMEM;
	entry = debugfs_create_file("pacing_gain_override", 0600, ctl_dir, NULL, &gain_fops);
	if (IS_ERR_OR_NULL(entry)) goto failed;
	entry = debugfs_create_file("cruise_active", 0400, ctl_dir, NULL, &cruise_fops);
	if (IS_ERR_OR_NULL(entry)) goto failed;
	entry = debugfs_create_file("status", 0400, ctl_dir, NULL, &status_fops);
	if (IS_ERR_OR_NULL(entry)) goto failed;
	return 0;
failed:
	debugfs_remove_recursive(ctl_dir);
	return entry ? PTR_ERR(entry) : -ENOMEM;
}

void bbr_qrl_ctl_exit(void)
{
	debugfs_remove_recursive(ctl_dir);
}
