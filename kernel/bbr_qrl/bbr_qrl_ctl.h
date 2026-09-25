/* SPDX-License-Identifier: GPL-2.0-only */
#ifndef BBR_QRL_CTL_H
#define BBR_QRL_CTL_H
struct sock;
int bbr_qrl_ctl_init(void);
void bbr_qrl_ctl_exit(void);
void bbr_qrl_attach(struct sock *sk);
void bbr_qrl_detach(struct sock *sk);
int bbr_qrl_apply(struct sock *sk, int phase, int eligible, int native_gain);
#endif
