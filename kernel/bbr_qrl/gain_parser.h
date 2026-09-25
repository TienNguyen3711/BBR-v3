/* SPDX-License-Identifier: GPL-2.0-only */
/* Shared by kernel code and compiled host tests. Caller supplies errno/size_t. */
#ifndef QBBR_GAIN_PARSER_H
#define QBBR_GAIN_PARSER_H
static inline int qbbr_space(char c)
{
	return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}
static inline int bbr_qrl_parse_gain(const char *buf, size_t len, int *out)
{
	size_t i = 0, end = len;
	int negative = 0, whole, fraction = 0, digits = 0, value;
	if (!len || len >= 32) return -EINVAL;
	for (i = 0; i < len; i++) if (!buf[i]) return -EINVAL;
	i = 0;
	while (i < end && qbbr_space(buf[i])) i++;
	while (end > i && qbbr_space(buf[end - 1])) end--;
	if (i == end) return -EINVAL;
	if (buf[i] == '-') { negative = 1; i++; }
	if (i == end || buf[i] < '0' || buf[i] > '9') return -EINVAL;
	whole = buf[i++] - '0';
	if (i < end) {
		if (buf[i++] != '.' || i == end) return -EINVAL;
		while (i < end) {
			if (buf[i] < '0' || buf[i] > '9' || digits == 6) return -EINVAL;
			fraction = fraction * 10 + buf[i++] - '0';
			digits++;
		}
	}
	while (digits++ < 6) fraction *= 10;
	if (negative) {
		if (whole != 1 || fraction != 0) return -EINVAL;
		*out = -1;
		return 0;
	}
	value = whole * 1000000 + fraction;
	switch (value) {
	case 750000: *out = 192; break;
	case 900000: *out = 230; break;
	case 1000000: *out = 256; break;
	case 1100000: *out = 282; break;
	case 1250000: *out = 320; break;
	default: return -ERANGE;
	}
	return 0;
}
#endif
