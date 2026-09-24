"""Compile and execute the actual kernel parser, rather than a Python mirror."""
from __future__ import annotations
import ctypes
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
ACTION_LEVELS = (0.75, 0.90, 1.00, 1.10, 1.25)
EXPECTED_GAINS = {0.75: 192, 0.90: 230, 1.00: 256, 1.10: 282, 1.25: 320}

@pytest.fixture(scope='module')
def c_parser(tmp_path_factory):
    work = tmp_path_factory.mktemp('actual-kernel-parser')
    source = work / 'parser.c'
    source.write_text('#include <stddef.h>\n#include <errno.h>\n#include "gain_parser.h"\n'
                      'int parse(const char *p, size_t n, int *out) { return bbr_qrl_parse_gain(p,n,out); }\n')
    lib = work / ('parser.dylib' if sys.platform == 'darwin' else 'parser.so')
    subprocess.run(['cc', '-std=c99', '-Wall', '-Wextra', '-Werror', '-fPIC',
                    '-dynamiclib' if sys.platform == 'darwin' else '-shared',
                    '-I', str(ROOT/'kernel/bbr_qrl'), str(source), '-o', str(lib)], check=True)
    fn = ctypes.CDLL(str(lib)).parse
    fn.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
    fn.restype = ctypes.c_int
    def parse(data):
        if isinstance(data, str): data = data.encode()
        out = ctypes.c_int(777)
        ret = fn(data, len(data), ctypes.byref(out))
        return ret, out.value
    return parse

@pytest.mark.parametrize('level', ACTION_LEVELS)
def test_exact_five_actions(c_parser, level):
    assert c_parser(f'{level:.6f}\n') == (0, EXPECTED_GAINS[level])

@pytest.mark.parametrize('text', ['-1', '-1.0\n', '  -1.000000  ', '\t1.100000\n', '0.9'])
def test_supported_forms(c_parser, text):
    ret, value = c_parser(text)
    assert ret == 0
    assert value in (-1, 230, 282)

@pytest.mark.parametrize('text', ['0.5', '0.74', '1.26', '2.0', '1.05', '1.125', '0.899',
    '-2', '-1.1', '-1garbage', '-0.9', '99999999999999999999999999999',
    '1.1000000', '1.1abc', '1.', '', ' ', '+1', '01.0', b'1.1\x00junk', b'\x00', '1'*32])
def test_bad_or_undeclared_inputs_do_not_modify_output(c_parser, text):
    ret, value = c_parser(text)
    assert ret < 0
    assert value == 777


def test_parser_accepts_only_declared_decimal_grid(c_parser):
    for micro in range(700000, 1300001, 100):
        ret, value = c_parser(f'{micro//1000000}.{micro%1000000:06d}')
        if micro in [750000,900000,1000000,1100000,1250000]:
            assert ret == 0
        else:
            assert ret < 0 and value == 777


def test_action_contract_still_matches():
    import yaml
    config = yaml.safe_load((ROOT/'qbbr/configs/action_pacing_gain.yaml').read_text())
    assert tuple(config['levels']) == ACTION_LEVELS
    assert config['clamp_state'] == 'ProbeBW_CRUISE'
