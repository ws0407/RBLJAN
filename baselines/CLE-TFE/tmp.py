from binascii import a2b_hex


def hex2palin(s: str):
    s1 = bytes(s, 'utf-8')
    s2 = a2b_hex(s1)
    print(s2.decode())


def plain2hex(s: str):
    return s.encode('utf-8').hex().upper()


def hex2int(s: str):
    return [int(s[i:i + 2], 16) for i in range(0, len(s), 2)]


if __name__ == '__main__':
    ss = 'Hello, Huawei!'
    res = plain2hex(ss)
    res = res + '0' * (256 - len(res))
    print(res)
    print(hex2int(res))