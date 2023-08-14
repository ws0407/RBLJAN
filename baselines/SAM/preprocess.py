import binascii
import numpy as np
import dpkt
import random
import pickle
import os, time
from scapy.all import *

protocols = ['dns', 'smtp', 'ssh', 'ftp', 'http', 'https']
ports = [53, 25, 22, 21, 80, 443]
# CLASSES = ['Cridex', 'Geodo', 'Htbot', 'Miuref', 'Neris', 'Nsis-ay', 'Shifu', 'Tinba', 'Virut', 'Zeus']
# CLASSES = ['amazon', 'baidu', 'bing', 'douban', 'facebook', 'google', 'imdb', 'instagram', 'iqiyi', 'jd',
#            'neteasemusic', 'qqmail', 'reddit', 'taobao', 'ted', 'tieba', 'twitter', 'weibo', 'youku', 'youtube']
# CLASSES = ['Cridex', 'Geodo', 'Htbot', 'Miuref', 'Neris', 'Nsis-ay', 'Shifu', 'Tinba', 'Virut', 'Zeus',
#            'BitTorrent', 'FTP', 'Facetime', 'Gmail', 'MySQL', 'Outlook', 'Skype', 'WorldOfWarcraft', 'SMB', 'Weibo']
# CLASSES = ['vimeo', 'spotify', 'voipbuster', 'sinauc', 'cloudmusic', 'weibo', 'baidu', 'tudou', 'amazon', 'thunder',
#            'gmail', 'pplive', 'qq', 'taobao', 'yahoomail', 'itunes', 'twitter', 'jd', 'sohu', 'youtube', 'youku',
#            'netflix', 'aimchat', 'kugou', 'skype', 'facebook', 'google', 'mssql', 'ms-exchange']
CLASSES = ['audio', 'chat', 'file', 'mail', 'streaming', 'voip',
           'vpn-audio', 'vpn-chat', 'vpn-file', 'vpn-mail', 'vpn-streaming', 'vpn-voip']
LABELS = {k: v for k, v in zip(CLASSES, range(len(CLASSES)))}

NGRAM = 50

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
        self.flag = False
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
            # print(file_names)
            # print(file_paths)
            for file_path, file_name in zip(file_paths, file_names):
                if os.path.isdir(file_path):
                    X = []
                    y = []
                    _s_t = time.time()
                    valid_num, invalid_num = 0, 0
                    label_index += 1
                    label = file_name
                    # if os.path.isfile('/data/ws/tmp/BLJAN-IWQOS/datasets/EBSNN/result_doc_sam/y_' + label + '.pkl'):
                    #     print('skip label: {}'.format(label))
                    #     continue
                    print('>> start process files of label: {}'.format(label))
                    pcap_files = [os.path.join(file_path, pcap_file) for pcap_file in os.listdir(file_path)]
                    for pcap_file in pcap_files:
                        if len(y) >= 100000:
                            break
                        if '.pcap' not in pcap_file:
                            continue
                        _valid_num, _invalid_num = 0, 0
                        print('>> >> start read pcap: {}'.format(pcap_file))
                        try:
                            pcap = dpkt.pcap.Reader(open(pcap_file, 'rb'))
                            self.flag = False
                        except Exception as e:
                            print('error in {}, error: {}'.format(pcap_file, e))
                            self.flag = True
                            # return
                        if self.flag or pcap.datalink() != dpkt.pcap.DLT_EN10MB:
                            print('unknow data link! {}'.format(label))
                            pcap = rdpcap(pcap_file)
                            print('rdpcap done!')
                            for i in range(len(pcap)):
                                try:
                                    payload = pcap[i].payload
                                    if 'vpn' in pcap_file:
                                        payload = pcap[i]
                                    if len(payload.payload.payload) <= 0:
                                        _invalid_num += 1
                                        continue
                                    version = payload.version
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
                                    if hasattr(payload.payload, 'ack'):
                                        payload.payload.ack = 0
                                    if hasattr(payload.payload, 'seq'):
                                        payload.payload.seq = 0
                                    if len(payload) >= NGRAM + 1:
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
                                        y.append(LABELS[label])
                                        _valid_num += 1
                                        if len(y) >= 100000:
                                            break
                                        # if len(y) >= 200000:
                                        #     break
                                    else:
                                        _invalid_num += 1
                                except Exception as e:
                                    # print('>> >> error: {} in packet {}, label: {}'.format(e, i, label))
                                    _invalid_num += 1
                        else:
                            for timestamp, buf in pcap:
                                try:
                                    eth = dpkt.ethernet.Ethernet(buf)
                                    # if isinstance(eth.data, dpkt.ip.IP):
                                    # tcp or udp packet
                                    ip = eth.data
                                    if len(ip) > NGRAM and len(ip.data.data) > 0:
                                        if ip.v == 4:
                                            # continue
                                            ip.src = b'\x00' * 4
                                            ip.dst = b'\x00' * 4
                                        elif ip.v == 6:
                                            ip.src = b'\x00' * 16
                                            ip.dst = b'\x00' * 16
                                        if hasattr(ip, 'sum'):
                                            ip.sum = 0
                                        if hasattr(ip, 'id'):
                                            ip.id = 0
                                        if hasattr(ip, 'offset'):
                                            ip.offset = 0

                                        ip.data.sport = 0
                                        ip.data.dport = 0
                                        if hasattr(ip.data, 'sum'):
                                            ip.data.sum = 0

                                        if isinstance(ip.data, dpkt.tcp.TCP):
                                            ip.data.seq = 0
                                            ip.data.ack = 0

                                        res = str(binascii.hexlify(bytes_encode(ip)).decode())
                                        if self.byte_len != -1:
                                            res = res[:2*self.byte_len]
                                        res = [''.join(x).upper() for x in zip(res[::2], res[1::2])]
                                        bytes_idx = [word_to_idx[w] for w in res]
                                        X.append(bytes_idx)
                                        y.append(LABELS[label])
                                        _valid_num += 1
                                        if len(y) >= 100000:
                                            break
                                        # if len(y) >= 200000:
                                        #     break
                                    else:
                                        # print('>> >> no payload in packet {}'.format(_invalid_num + _valid_num))
                                        _invalid_num += 1
                                except Exception as e:
                                    # print('>> >> error: {} in packet {}, label: {}'.format(e, _valid_num + _invalid_num, label))
                                    invalid_num += 1
                        invalid_num += _invalid_num
                        valid_num += _valid_num
                    invalid_total += invalid_num
                    valid_total += valid_num
                    print(
                        '>> process files of label: {} done with {}s, valid packets: {}. skip {} (invalid) packets'.format(
                            label, time.time() - _s_t, valid_num, invalid_num))

                    _s_t = time.time()
                    with open(self.file_out_dir + 'X_' + label + '.pkl', 'wb') as f1:
                        pickle.dump(X, f1)
                    print('dump X cost {}s'.format(time.time() - _s_t))
                    _s_t = time.time()
                    with open(self.file_out_dir + 'y_' + label + '.pkl', 'wb') as f2:
                        pickle.dump(y, f2)
                    print('dump y cost {}s'.format(time.time() - _s_t))

        print('All done with {}s, valid packets: {}. skip {} (invalid) packets'.format(
            time.time() - s_t_out, valid_total, invalid_total))
        _s_t = time.time()


def gen_flows(pcap):
    flows = [{} for _ in range(len(protocols))]

    if pcap.datalink() != dpkt.pcap.DLT_EN10MB:
        print('unknow data link!')
        return

    xgr = 0
    for _, buff in pcap:
        eth = dpkt.ethernet.Ethernet(buff)
        xgr += 1
        if xgr % 500000 == 0:
            print('The %dth pkt!' % xgr)
        # break

        if isinstance(eth.data, dpkt.ip.IP) and (
                isinstance(eth.data.data, dpkt.udp.UDP)
                or isinstance(eth.data.data, dpkt.tcp.TCP)):
            # tcp or udp packet
            ip = eth.data

            # loop all protocols
            for name in protocols:
                index = protocols.index(name)
                if ip.data.sport == ports[index] or \
                        ip.data.dport == ports[index]:
                    if len(flows[index]) >= 10000:
                        # each class has at most 1w flows
                        break
                    # match a protocol
                    key = '.'.join(map(str, map(int, ip.src))) + \
                          '.' + '.'.join(map(str, map(int, ip.dst))) + \
                          '.' + '.'.join(map(str, [ip.p, ip.data.sport, ip.data.dport]))

                    if key not in flows[index]:
                        flows[index][key] = [ip]
                    elif len(flows[index][key]) < 1000:
                        # each flow has at most 1k flows
                        flows[index][key].append(ip)
                    # after match a protocol quit
                    break

    return flows


# def split_train_test(flows, name, k):
# 	keys = list(flows.keys())

# 	test_keys = keys[k*int(len(keys)*0.1):(k+1)*int(len(keys)*0.1)]
# 	test_min = 0xFFFFFFFF
# 	test_flows = {}
# 	for k in test_keys:
# 		test_flows[k] = flows[k]
# 		test_min = min(test_min, len(flows[k]))

# 	train_keys = set(keys) - set(test_keys)
# 	train_min = 0xFFFFFFFF
# 	train_flows = {}
# 	for k in train_keys:
# 		train_flows[k] = flows[k]
# 		train_min = min(train_min, len(flows[k]))

# 	print('============================')
# 	print('Generate flows for %s'%name)
# 	print('Total flows: ', len(flows))
# 	print('Train flows: ', len(train_flows), ' Min pkts: ', train_min)
# 	print('Test flows: ', len(test_flows), ' Min pkts: ', test_min)

# 	return train_flows, test_flows


def closure(flows):
    flow_dict = {}
    for name in protocols:
        index = protocols.index(name)
        flow_dict[name] = flows[index]
        print('============================')
        print('Generate flows for %s' % name)
        print('Total flows: ', len(flows[index]))
        cnt = 0
        for k, v in flows[index].items():
            cnt += len(v)
        print('Total pkts: ', cnt)

    with open('pro_flows.pkl', 'wb') as f:
        pickle.dump(flow_dict, f)


if __name__ == '__main__':
    # pcap = dpkt.pcap.Reader(open('/data/xgr/sketch_data/wide/202006101400.pcap', 'rb'))
    # flows = gen_flows(pcap)
    # closure(flows)

    s_t_main = time.time()

    _file_in_dir = ['./data/ISCX-VPN/dataset/']
    _file_out_dir = './data/ISCX-VPN/result_doc_sam/'

    traffic = PCAP_DPKT(_file_in_dir, _file_out_dir, byte_len=50)
    # traffic.extract_flows()
    traffic.get_idx_saved()
    # traffic.get_protocols()

    print('\\data_preprocessing.py finished with {}s'.format(time.time() - s_t_main))

