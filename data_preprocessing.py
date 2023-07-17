#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import dpkt
import random
import pickle

import binascii, pickle, os
from dpkt.utils import inet_to_str
from scapy.all import *
# from utils import *
NGRAM = 50
PKT_MAX_LEN = 1500


class PCAP_DPKT:
    def __init__(self, _file_in_dirs, _file_out_dir, flow_len=-1, byte_len=-1,
                 flow_flag=False, uniform_byte_len=False, flow_min_len=2,
                 _byte_min_len=-1):
        self.file_in_dirs = _file_in_dirs
        self.file_out_dir = _file_out_dir
        self.flow_len = flow_len
        self.byte_len = byte_len
        self.flow_flag = flow_flag
        self.uniform_byte_len = uniform_byte_len
        self.flow_min_len = flow_min_len
        self.byte_min_len = _byte_min_len
        if not os.path.exists(self.file_out_dir):
            os.makedirs(self.file_out_dir)

    def get_idx_saved(self):
        invalid_total = 0
        valid_total = 0
        label_index = 0
        s_t_out = time.time()
        print('start process pcap to hex words files')
        word_to_idx = {(hex(i)[2:].zfill(2)).upper(): i for i in range(256)}
        for file_in_dir in self.file_in_dirs:
            file_names = os.listdir(file_in_dir)
            file_paths = [os.path.join(file_in_dir, file_name) for file_name in file_names]
            # print(file_names, file_paths)
            for file_path, file_name in zip(file_paths, file_names):
                if os.path.isdir(file_path):
                    X = []
                    _s_t = time.time()
                    valid_num, invalid_num = 0, 0   # of every label
                    label_index += 1
                    label = file_name.lower()
                    print('>> start process files of label: {}'.format(label))
                    pcap_files = [os.path.join(file_path, pcap_file) for pcap_file in os.listdir(file_path)]
                    for pcap_file in pcap_files:
                        _valid_num, _invalid_num = 0, 0     # of every pcap file (one label may contain many pcap file)
                        print('>> >> start read pcap: {}'.format(pcap_file))
                        try:
                            pcap = dpkt.pcap.Reader(open(pcap_file, 'rb'))
                            print('>> >> read done')
                        except Exception as e:
                            pcap = None
                            print('[!] error in {}, error: {}'.format(pcap_file, e))
                        if pcap is None or pcap.datalink() != dpkt.pcap.DLT_EN10MB:
                            """dpkt is very fast when reading and processing, but only supports specific formats
                               when dpkt goes wrong, use scapy! slow..., but very strong!"""
                            print('[!] unknown data link! trying by scapy.rdpcap()...')
                            pcap = rdpcap(pcap_file)
                            print('>> >> read done!')
                            for i in range(len(pcap)):
                                try:
                                    # if len(X) >= 100000:
                                    #     break
                                    payload = pcap[i]
                                    if 'vpn' not in pcap_file:  # vpn traffic has no Ethernet header
                                        payload = pcap[i].payload
                                    version = payload.version
                                    if len(payload.payload.payload) <= 0 or len(payload) <= NGRAM:
                                        _invalid_num += 1
                                        continue
                                    sport = payload.payload.sport
                                    dport = payload.payload.dport
                                    # some very nasty packets, whose payload has nothing useful!
                                    if sport == 5353 or dport == 5353 or sport == 137 or sport == 138 or sport == 5355 or dport == 5355:
                                        # print('skip ip address relative pkt') #  MDNS,
                                        _invalid_num += 1
                                        continue
                                    # if sport == 161 or dport == 161:
                                    #     # print('skip SNMP pkt')
                                    #     _invalid_num += 1
                                    #     continue
                                    if sport == 17500 or dport == 17500 or sport == 1900 or dport == 1900:
                                        # print('skip db-lsp-disc/ssdp pkt')
                                        _invalid_num += 1
                                        continue
                                    if sport == 57621 or dport == 57621 and label == 'mail':
                                        # print('skip broadcast pkt')
                                        _invalid_num += 1
                                        continue
                                    if sport == 4644 or dport == 4644 and label == 'chat':
                                        # print('skip broadcast pkt')
                                        _invalid_num += 1
                                        continue
                                    if hasattr(payload.payload, 'flags') and hasattr(payload.payload.flags, 'value'):
                                        if payload.payload.flags.value % 2 == 1:
                                            # print('skip FIN pkt')
                                            _invalid_num += 1
                                            continue
                                        if payload.payload.flags.value % 4 == 2:
                                            # print('skip SYN pkt')
                                            _invalid_num += 1
                                            continue
                                    if hasattr(payload, 'chksum'):
                                        payload.chksum = 0
                                    if hasattr(payload, 'id'):
                                        payload.id = 0
                                    if hasattr(payload, 'offset'):
                                        payload.offset = 0
                                    if hasattr(payload.payload, 'chksum'):
                                        payload.payload.chksum = 0
                                    payload.payload.sport = 0
                                    payload.payload.dport = 0
                                    # if payload.payload.sport > 10000:
                                    #     payload.payload.sport = 0
                                    # if payload.payload.dport > 10000:
                                    #     payload.payload.dport = 0

                                    if hasattr(payload.payload, 'ack'):
                                        payload.payload.ack = 0
                                    if hasattr(payload.payload, 'seq'):
                                        payload.payload.seq = 0
                                    res = str(binascii.hexlify(bytes_encode(payload)).decode())
                                    if self.byte_len != -1:
                                        res = res[:2 * self.byte_len]
                                    res = [''.join(x).upper() for x in zip(res[::2], res[1::2])]
                                    if version == 4:
                                        res[12:20] = ['00'] * 8
                                    elif version == 6:
                                        res[8:40] = ['00'] * 32
                                    bytes_idx = [word_to_idx[w] for w in res]
                                    X.append(bytes_idx)
                                    _valid_num += 1
                                except Exception as e:
                                    # print('>> >> error: {} in packet {}, label: {}'.format(e, i, label))
                                    _invalid_num += 1
                        else:
                            """process by dpkt, a similar way to scapy above"""
                            for timestamp, buf in pcap:
                                try:
                                    eth = dpkt.ethernet.Ethernet(buf)
                                    # if len(X) >= 100000:
                                    #     break
                                    # if isinstance(eth.data, dpkt.ip.IP):
                                    # tcp or udp packet
                                    ip = eth.data
                                    v = ip.v
                                    if len(ip) <= NGRAM or len(ip.data.data) <= 0:
                                        _invalid_num += 1
                                        continue
                                    if hasattr(ip.data, 'flags'):
                                        if ip.data.flags % 2 == 1:
                                            # print('skip FIN pkt')
                                            _invalid_num += 1
                                            continue
                                        if ip.data.flags % 4 == 2:
                                            # print('skip SYN pkt')
                                            _invalid_num += 1
                                            continue
                                    sport = ip.data.sport
                                    dport = ip.data.dport
                                    if sport == 5353 or dport == 5353 or sport == 137 or sport == 138 or sport == 5355 or dport == 5355:
                                        # print('skip MDNS or NBNS pkt')
                                        _invalid_num += 1
                                        continue
                                    # if sport == 161 or dport == 161:
                                    #     # print('skip MDNS or NBNS pkt')
                                    #     _invalid_num += 1
                                    #     continue
                                    if sport == 17500 or dport == 17500 or sport == 1900 or dport == 1900:
                                        # print('skip db-lsp-disc/ssdp pkt')
                                        _invalid_num += 1
                                        continue
                                    if sport == 57621 or dport == 57621 and label == 'mail':
                                        # print('skip broadcast pkt')
                                        _invalid_num += 1
                                        continue
                                    if sport == 4644 or dport == 4644 and label == 'chat':
                                        # print('skip broadcast pkt')
                                        _invalid_num += 1
                                        continue
                                    # if ip.src == b'\xff\xff\xff\xff' or ip.dst == b'\xff\xff\xff\xff':
                                    #     # print('skip broadcast pkt')
                                    #     _invalid_num += 1
                                    #     continue
                                    if v == 4:
                                        ip.src = b'\x00' * 4
                                        ip.dst = b'\x00' * 4
                                    elif v == 6:
                                        ip.src = b'\x00' * 16
                                        ip.dst = b'\x00' * 16
                                    if hasattr(ip, 'id'):
                                        ip.id = 0
                                    if hasattr(ip, 'offset'):
                                        ip.offset = 0
                                    ip.data.sport = 0
                                    ip.data.dport = 0
                                    # if ip.data.sport > 10000:
                                    #     ip.data.sport = 0
                                    # if ip.data.dport > 10000:
                                    #     ip.data.dport = 0
                                    if hasattr(ip, 'sum'):
                                        ip.sum = 0
                                    if hasattr(ip.data, 'sum'):
                                        ip.data.sum = 0
                                    if isinstance(ip.data, dpkt.tcp.TCP):
                                        ip.data.seq = 0
                                        ip.data.ack = 0
                                    res = str(binascii.hexlify(bytes_encode(ip)).decode())
                                    if self.byte_len != -1:
                                        res = res[:2 * self.byte_len]
                                    res = [''.join(x).upper() for x in zip(res[::2], res[1::2])]
                                    bytes_idx = [word_to_idx[w] for w in res]
                                    X.append(bytes_idx)
                                    _valid_num += 1
                                except Exception as e:
                                    # print('>> >> error: {} in packet {}, label: {}'.format(e, _valid_num + _invalid_num, label))
                                    invalid_num += 1
                        invalid_num += _invalid_num
                        valid_num += _valid_num
                        # print('valid packets: {}. skip {} (invalid) packets in file: {}'.format(_valid_num, _invalid_num, pcap_file))
                    invalid_total += invalid_num
                    valid_total += valid_num
                    print('>> processed files of label: {} done with {}s, valid packets: {}. skip {} (invalid) packets'.format(
                            label, time.time() - _s_t, valid_num, invalid_num))

                    _s_t = time.time()
                    with open(self.file_out_dir + label + '.pkl', 'wb') as f1:
                        pickle.dump(X, f1)
                    print('dump cost {}s'.format(time.time() - _s_t))

        print('All done with {}s, valid packets: {}. skip {} (invalid) packets'.format(
            time.time() - s_t_out, valid_total, invalid_total))
        _s_t = time.time()


    def count_flows(self):
        pass


    def get_flows(self):
        invalid_total = 0
        valid_total = 0
        label_index = 0
        s_t_get_flows = time.time()
        print('start extract flows from pcap file')
        word_to_idx = {(hex(i)[2:].zfill(2)).upper(): i for i in range(256)}
        for file_in_dir in self.file_in_dirs:
            file_names = os.listdir(file_in_dir)
            file_paths = [os.path.join(file_in_dir, file_name) for file_name in file_names]
            # print(file_names)
            # print(file_paths)
            for file_path, file_name in zip(file_paths, file_names):
                if not os.path.isdir(file_path):
                    continue
                flows = {}
                s_t_one_label = time.time()
                valid_label, invalid_label = 0, 0
                label_index += 1
                label = file_name.lower()
                print('>> start process files of label: {}'.format(label))
                pcap_files = [os.path.join(file_path, pcap_file) for pcap_file in os.listdir(file_path)]
                for pcap_file in pcap_files:
                    # if 'vimeo4.pcap' not in pcap_file:
                    #     continue
                    valid_pcap, invalid_pcap = 0, 0
                    print('>> >> start read pcap: {}'.format(pcap_file))
                    try:
                        pcap = dpkt.pcap.Reader(open(pcap_file, 'rb'))
                        print('>> >> read done')
                    except Exception as e:
                        pcap = None
                        print('[!] error in {}, error: {}'.format(pcap_file, e))
                    if pcap is None or pcap.datalink() != dpkt.pcap.DLT_EN10MB:
                        print('[!] unknown data link! try to read by scapy.rdpcap()')
                        pcap = rdpcap(pcap_file)
                        print('>> >> read done!')
                        for i in range(len(pcap)):
                            try:
                                payload = pcap[i]
                                if 'vpn' not in pcap_file:
                                    payload = pcap[i].payload
                                version = payload.version
                                if len(payload.payload.payload) <= 0 or len(payload) <= NGRAM:
                                    invalid_pcap += 1
                                    continue
                                src = payload.src
                                dst = payload.dst
                                sport = payload.payload.sport
                                dport = payload.payload.dport
                                if sport == 5353 or sport == 137 or sport == 138 or dport == 5353 or dport == 137 or dport == 138:
                                    # print('skip ip address relative pkt')
                                    invalid_pcap += 1
                                    continue
                                if hasattr(payload.payload, 'flags') and hasattr(payload.payload.flags, 'value'):
                                    if payload.payload.flags.value % 2 == 1:
                                        # print('skip FIN pkt')
                                        invalid_pcap += 1
                                        continue
                                    if payload.payload.flags.value % 4 == 2:
                                        # print('skip SYN pkt')
                                        invalid_pcap += 1
                                        continue
                                proto = payload.proto
                                type_a = (src, dst, sport, dport, proto, pcap_file)  # a and b belong to the same flow
                                type_b = (dst, src, dport, sport, proto, pcap_file)  #

                                if type_a in flows:     # select the first 10 packets in the flow
                                    if len(flows[type_a]) >= 10:
                                        continue
                                if type_b in flows:
                                    if len(flows[type_b]) >= 10:
                                        continue

                                if hasattr(payload, 'chksum'):
                                    payload.chksum = 0
                                if hasattr(payload, 'id'):
                                    payload.id = 0
                                if hasattr(payload, 'offset'):
                                    payload.offset = 0
                                if hasattr(payload.payload, 'chksum'):
                                    payload.payload.chksum = 0
                                # if payload.payload.sport > 10000:
                                #     payload.payload.sport = 0
                                # if payload.payload.dport > 10000:
                                #     payload.payload.dport = 0
                                payload.payload.sport = 0
                                payload.payload.dport = 0
                                if hasattr(payload.payload, 'ack'):
                                    payload.payload.ack = 0
                                if hasattr(payload.payload, 'seq'):
                                    payload.payload.seq = 0
                                res = str(binascii.hexlify(bytes_encode(payload)).decode())
                                if self.byte_len != -1:
                                    res = res[:2 * self.byte_len]
                                res = [''.join(x).upper() for x in zip(res[::2], res[1::2])]
                                res = [word_to_idx[w] for w in res]
                                paddings = 255 if type_b in flows else 0
                                if version == 4:
                                    res[12:20] = [paddings] * 8
                                elif version == 6:
                                    res[8:40] = [paddings] * 32

                                # type_a and b belong to the same flow and share the same key in the dictionary
                                if type_a in flows:
                                    flows[type_a].append(res)
                                elif type_b in flows:
                                    flows[type_b].append(res)
                                else:
                                    flows[type_a] = [res]
                                valid_pcap += 1
                            except Exception as e:
                                # print('>> >> error: {} in packet {}, label: {}'.format(e, i, label))
                                invalid_pcap += 1
                    else:
                        for timestamp, buf in pcap:
                            try:
                                eth = dpkt.ethernet.Ethernet(buf)
                                ip = eth.data
                                v = ip.v
                                if len(ip) <= NGRAM or len(ip.data.data) <= 0:
                                    invalid_pcap += 1
                                    continue
                                src = ip.src
                                dst = ip.dst
                                sport = ip.data.sport
                                dport = ip.data.dport
                                if sport == 5353 or sport == 137 or sport == 138 or dport == 5353 or dport == 137 or dport == 138:
                                    # print('skip MDNS or NBNS pkt')
                                    invalid_pcap += 1
                                    continue
                                proto = ip.p
                                type_a = (src, dst, sport, dport, proto, pcap_file)
                                type_b = (dst, src, dport, sport, proto, pcap_file)

                                if type_a in flows:
                                    if len(flows[type_a]) >= 10:
                                        continue
                                if type_b in flows:
                                    if len(flows[type_b]) >= 10:
                                        continue

                                if hasattr(ip.data, 'flags'):
                                    if ip.data.flags % 2 == 1:
                                        # print('skip FIN pkt')
                                        invalid_pcap += 1
                                        continue
                                    if ip.data.flags % 4 == 2:
                                        # print('skip SYN pkt')
                                        invalid_pcap += 1
                                        continue
                                paddings = b'\xff' if type_b in flows else b'\x00'
                                if v == 4:
                                    ip.src = paddings * 4
                                    ip.dst = paddings * 4
                                elif v == 6:
                                    ip.src = paddings * 16
                                    ip.dst = paddings * 16
                                if hasattr(ip, 'id'):
                                    ip.id = 0
                                if hasattr(ip, 'offset'):
                                    ip.offset = 0
                                # if ip.data.sport > 10000:
                                #     ip.data.sport = 0
                                # if ip.data.dport > 10000:
                                #     ip.data.dport = 0
                                ip.data.sport = 0
                                ip.data.dport = 0
                                if hasattr(ip, 'sum'):
                                    ip.sum = 0
                                if hasattr(ip.data, 'sum'):
                                    ip.data.sum = 0
                                if isinstance(ip.data, dpkt.tcp.TCP):
                                    ip.data.seq = 0
                                    ip.data.ack = 0
                                res = str(binascii.hexlify(bytes_encode(ip)).decode())
                                if self.byte_len != -1:
                                    res = res[:2 * self.byte_len]
                                res = [''.join(x).upper() for x in zip(res[::2], res[1::2])]
                                res = [word_to_idx[w] for w in res]
                                if type_a in flows:
                                    flows[type_a].append(res)
                                elif type_b in flows:
                                    flows[type_b].append(res)
                                else:
                                    flows[type_a] = [res]
                                valid_pcap += 1

                            except Exception as e:
                                # print('>> >> error: {} in packet {}, label: {}'.format(e, valid_pcap + invalid_pcap, label))
                                invalid_pcap += 1
                    for key in flows.keys():    # exclude flows that only have few packets
                        if len(flows[key]) < 2:
                            valid_pcap -= len(flows[key])
                            flows[key] = []
                    # for key in flows:
                    #     if len(flows[key]) == 2:
                    #         (src, dst, sport, dport, proto, pcap_file) = key
                    #         if sport == 53 or dport == 53:
                    #             flows[key] = []
                    #             invalid_pcap += 2
                    #             valid_pcap -= 2
                    invalid_label += invalid_pcap
                    valid_label += valid_pcap
                    print('valid packets: {}. skip {} (invalid) packets in file: {}'.format(
                        valid_pcap, invalid_pcap, pcap_file[pcap_file.rfind('/') + 1:]))
                invalid_total += invalid_label
                valid_total += valid_label

                print('\n>> process files of label: {} done with {}s, flows: {}\n '
                      'valid packets: {}. skip {} (invalid) packets'.format(
                        label, time.time() - s_t_one_label, len(flows), valid_label, invalid_label))
                s_t_one_label = time.time()
                if len(flows) <= 300:
                    print("packet number <= 300, skip!")
                    continue
                with open(self.file_out_dir + label + '.pkl', 'wb') as f1:
                    pickle.dump(flows, f1)
                print('dump flows cost {}s\n'.format(time.time() - s_t_one_label))


if __name__ == '__main__':
    s_t_main = time.time()

    _file_in_dir = ['./datasets/X-App/']
    _file_out_dir = './datasets/X-App_flow/'

    traffic = PCAP_DPKT(_file_in_dir, _file_out_dir, byte_len=PKT_MAX_LEN)
    # traffic.get_idx_saved()
    traffic.get_flows()
    print('\ndata_preprocessing.py finished with {}s'.format(time.time() - s_t_main))
