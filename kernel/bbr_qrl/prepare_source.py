"""Generate a separately named BBR-v3 lab module from the exact pinned source."""
import argparse
import difflib
import hashlib
from pathlib import Path

PIN = 'b16a9a18178b55103fbd2028967952db52d19f36'
SOURCE_SHA256 = '6d88372b861cfbabb560dc65c4341933ebfd89f33a65df67ec9b1f0dd667b062'


def generate(source):
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Source does not match pinned BBR-v3; refusing a best-effort patch')
    text = source
    def change(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError('Patch anchor is not unique: ' + old[:60])
        text = text.replace(old, new, 1)
    change('#include "tcp_dctcp.h"', '#include "tcp_dctcp.h"\n#include "bbr_qrl_ctl.h"')
    change('out:\n\tbbr_advance_latest_delivery_signals(sk, rs, &ctx);', '''out:
	/* The lab controller runs even when the model takes its fast path.
	 * Only pacing_gain changes; phase transitions/cwnd/loss logic are stock.
	 * Native CRUISE commands latch on entry; clear/expiry restore on ACK. */
	{
		int phase = bbr->mode == BBR_PROBE_BW ? 4 + bbr->cycle_idx : bbr->mode;
		int native_gain = bbr->mode == BBR_PROBE_BW ?
			bbr_pacing_gain[bbr->cycle_idx] : bbr->pacing_gain;
		int eligible = bbr->mode == BBR_PROBE_BW &&
			bbr->cycle_idx == BBR_BW_PROBE_CRUISE &&
			inet_csk(sk)->icsk_ca_state == TCP_CA_Open &&
			rs->lost == 0 && rs->delivered_ce == 0;
		int gain = bbr_qrl_apply(sk, phase, eligible, native_gain);

		if (gain != bbr->pacing_gain) {
			bbr->pacing_gain = gain;
			bbr_set_pacing_rate(sk, bbr_bw(sk), gain);
		}
	}
	bbr_advance_latest_delivery_signals(sk, rs, &ctx);''')
    change('\tbbr->initialized = 1;', '\tbbr_qrl_attach(sk);\n\tbbr->initialized = 1;')
    change('\t.name\t\t= "bbr",', '\t.name\t\t= "bbr_qrl",')
    change('\t.init\t\t= bbr_init,', '\t.init\t\t= bbr_init,\n\t.release\t= bbr_qrl_release,')
    change('static struct tcp_congestion_ops tcp_bbr_cong_ops __read_mostly = {', 'static void bbr_qrl_release(struct sock *sk)\n{\n\tstruct bbr *bbr = inet_csk_ca(sk);\n\t/* TCP may release an algorithm selected before connect, without init. */\n\tif (bbr->initialized) {\n\t\tbbr->initialized = 0;\n\t\tbbr_qrl_detach(sk);\n\t}\n}\n\nstatic struct tcp_congestion_ops tcp_bbr_cong_ops __read_mostly = {')
    # The separately named laboratory copy is not a BPF struct-ops provider.
    start = text.index('BTF_SET8_START(tcp_bbr_check_kfunc_ids)')
    end = text.index('static int __init bbr_register(void)', start)
    text = text[:start] + text[end:]
    change('''	ret = register_btf_kfunc_id_set(BPF_PROG_TYPE_STRUCT_OPS, &tcp_bbr_kfunc_set);
	if (ret < 0)
		return ret;
	return tcp_register_congestion_control(&tcp_bbr_cong_ops);''', '''	ret = bbr_qrl_ctl_init();
	if (ret) return ret;
	ret = tcp_register_congestion_control(&tcp_bbr_cong_ops);
	if (ret) bbr_qrl_ctl_exit();
	return ret;''')
    change('\ttcp_unregister_congestion_control(&tcp_bbr_cong_ops);',
           '\ttcp_unregister_congestion_control(&tcp_bbr_cong_ops);\n\tbbr_qrl_ctl_exit();')
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--patch', type=Path)
    args = ap.parse_args()
    source = args.source.read_text()
    generated = generate(source)
    args.out.write_text(generated)
    if args.patch:
        args.patch.write_text(''.join(difflib.unified_diff(source.splitlines(True), generated.splitlines(True),
            fromfile='a/net/ipv4/tcp_bbr.c', tofile='b/net/ipv4/tcp_bbr.c')))
    print('Verified source ' + SOURCE_SHA256 + '; generated ' + str(args.out))

if __name__ == '__main__':
    main()
