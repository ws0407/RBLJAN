import argparse
import numpy as np
from scapy.all import *

from utils import show_time
from config import *
from tqdm import tqdm


def split_npz(dir_path_dict):
    for category in dir_path_dict:
        dir_path = dir_path_dict[category]
        file_list = os.listdir(dir_path)
        for file in file_list:
            if not file.endswith('.npz'):
                continue
            file_path = dir_path + '/' + file
            print('{} {} Process Starting'.format(show_time(), file_path))
            flows = {}
            npz = np.load(file_path, allow_pickle=True)

            header = npz['header']
            payload = npz['payload']
            payload_length = npz['payload_length']
            pkt_length = npz['pkt_length']
            src_ip = npz['src_ip']
            dst_ip = npz['dst_ip']
            src_port = npz['src_port']
            dst_port = npz['dst_port']
            time = npz['time']
            protocol = npz['protocol']
            flag = npz['flag']
            mss = npz['mss']

            pkt_num = len(npz['header'])
            for i in tqdm(range(pkt_num)):
                if len(payload[i]) == 0 and len(header[i]) < 40:
                    continue
                src = src_ip[i]
                dst = dst_ip[i]
                sport = src_port[i]
                dport = dst_port[i]
                proto = protocol[i]
                type_a = (src, dst, sport, dport, proto)  # a and b belong to the same flow
                type_b = (dst, src, dport, sport, proto)  #
                if type_a in flows:
                    flows[type_a].append(i)
                elif type_b in flows:
                    flows[type_b].append(i)
                else:
                    flows[type_a] = [i]
            total = 0
            for k, v in flows.items():
                if len(v) <= 1:
                    continue
                save_file = '{}_{}_{}_{}_{}_{}_{}.npz'.format(file[:-4], len(v), k[0], k[1], k[2], k[3], k[4])
                save_file = save_file.replace(':', '.')
                np.savez_compressed(dir_path + '/' + save_file,
                                    header=np.array([header[_] for _ in v], dtype=object),
                                    payload=np.array([payload[_] for _ in v], dtype=object),
                                    payload_length=np.array([payload_length[_] for _ in v], dtype=object),
                                    pkt_length=np.array([pkt_length[_] for _ in v], dtype=object),
                                    src_ip=np.array([src_ip[_] for _ in v], dtype=object),
                                    dst_ip=np.array([dst_ip[_] for _ in v], dtype=object),
                                    src_port=np.array([src_port[_] for _ in v], dtype=object),
                                    dst_port=np.array([dst_port[_] for _ in v], dtype=object),
                                    time=np.array([time[_] for _ in v], dtype=object),
                                    protocol=np.array([protocol[_] for _ in v], dtype=object),
                                    flag=np.array([flag[_] for _ in v], dtype=object),
                                    mss=np.array([mss[_] for _ in v], dtype=object))
                total += 1
            print('[{}] total flows: {}'.format(file[file.rfind('/'):-4], total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    opt = parser.parse_args()

    if opt.dataset == 'iscx-vpn':
        config = ISCXVPNConfig()
    elif opt.dataset == 'iscx-nonvpn':
        config = ISCXNonVPNConfig()
    elif opt.dataset == 'iscx-tor':
        config = ISCXTorConfig()
    elif opt.dataset == 'iscx-nontor':
        config = ISCXNonTorConfig()
    elif opt.dataset == 'ustc-malware':
        config = USTCMalwareConfig()
    else:
        raise Exception('Dataset Error')

    split_npz(config.DIR_PATH_DICT)
