# Data PrePrecessing

## Download

download the dataset from the link below or select another traffic dataset

- [X-APP](https://drive.google.com/file/d/1C-K9V03plCPrv5k3lvrwVLxCkm-WPlk5/view?usp=drive_link)
- [X-WEB](https://drive.google.com/file/d/1S_4Z1i5vwU3nFya08UYlrvpUFFkl-r1f/view?usp=drive_link)
- [USTC-TFC](https://github.com/yungshenglu/USTC-TFC2016)
- [ISCX-VPN](https://www.unb.ca/cic/datasets/vpn.html)

## Organize File directory

* extract the data into a folder (e.g., `X-APP`) under `./data/`
* the subfolders of the `./data/X-APP/` are all named candidate labels (e.g., `./data/X-APP/Viemo/`)
* `./data/X-APP/Viemo/` contains all PCAP files of this label

```python
data/
  ├── dataset1
  │      ├── label_1
  │      │    ├── f1.pcap
  │      │    ├── ...
  │      │    └── fn.pcap
  │      ├── label_2
  │      │    └── f.pcap
  │      ├── ...
  │      └── label_n
  │           ├── f1.pcap
  │           ├── ...
  │           └── fn.pcap
  ├── dataset2
  │      ├── label_1
  │      │    ├── f1.pcap
  │      ...
  │   ...   
  └── data_preprocessing.md
```

## Preprocessing

* set the parameter `CLASSES` in `utils.py` according to your requirement. If you choose an other dataset, modify the parameter `DATA_MAP` to specify the file directory
* run `data_preprocessing.py` to process the data into `.pkl` files
* The preprocessed traffic files are output at `./data/{dataset}_pkt/{label}.pkl`


## Appendix

My ./data/ directory tree

```python
./data/
├── ISCX-VPN
│   ├── audio
│   │   ├── facebook_audio1a.pcap
│   │   ├── facebook_audio1b.pcap
│   │   ├── facebook_audio2a.pcap
│   │   ├── facebook_audio2b.pcap
│   │   ├── facebook_audio3.pcap
│   │   ├── facebook_audio4.pcap
│   │   ├── hangouts_audio1a.pcap
│   │   ├── hangouts_audio1b.pcap
│   │   ├── hangouts_audio2a.pcap
│   │   ├── hangouts_audio2b.pcap
│   │   ├── hangouts_audio3.pcap
│   │   ├── hangouts_audio4.pcap
│   │   ├── skype_audio1a.pcap
│   │   ├── skype_audio1b.pcap
│   │   ├── skype_audio2a.pcap
│   │   ├── skype_audio2b.pcap
│   │   ├── skype_audio3.pcap
│   │   └── skype_audio4.pcap
│   ├── chat
│   │   ├── aimchat1.pcap
│   │   ├── aimchat2.pcap
│   │   ├── aim_chat_3a.pcap
│   │   ├── aim_chat_3b.pcap
│   │   ├── facebookchat1.pcap
│   │   ├── facebookchat2.pcap
│   │   ├── facebookchat3.pcap
│   │   ├── facebook_chat_4a.pcap
│   │   ├── facebook_chat_4b.pcap
│   │   ├── hangout_chat_4b.pcap
│   │   ├── hangouts_chat_4a.pcap
│   │   ├── ICQchat1.pcap
│   │   ├── ICQchat2.pcap
│   │   ├── icq_chat_3a.pcap
│   │   ├── icq_chat_3b.pcap
│   │   ├── skype_chat1a.pcap
│   │   └── skype_chat1b.pcap
│   ├── file
│   │   ├── ftps_down_1a.pcap
│   │   ├── ftps_down_1b.pcap
│   │   ├── ftps_up_2a.pcap
│   │   ├── ftps_up_2b.pcap
│   │   ├── scp1.pcap
│   │   ├── scpDown1.pcap
│   │   ├── scpDown2.pcap
│   │   ├── scpDown3.pcap
│   │   ├── scpDown4.pcap
│   │   ├── scpDown5.pcap
│   │   ├── scpDown6.pcap
│   │   ├── scpUp1.pcap
│   │   ├── scpUp2.pcap
│   │   ├── scpUp3.pcap
│   │   ├── scpUp5.pcap
│   │   ├── scpUp6.pcap
│   │   ├── sftp1.pcap
│   │   ├── sftpDown1.pcap
│   │   ├── sftpDown2.pcap
│   │   ├── sftp_down_3a.pcap
│   │   ├── sftp_down_3b.pcap
│   │   ├── sftpUp1.pcap
│   │   ├── sftp_up_2a.pcap
│   │   ├── sftp_up_2b.pcap
│   │   ├── skype_file1.pcap
│   │   ├── skype_file2.pcap
│   │   ├── skype_file3.pcap
│   │   ├── skype_file4.pcap
│   │   ├── skype_file5.pcap
│   │   ├── skype_file6.pcap
│   │   ├── skype_file7.pcap
│   │   └── skype_file8.pcap
│   ├── mail
│   │   ├── email1a.pcap
│   │   ├── email1b.pcap
│   │   ├── email2a.pcap
│   │   ├── email2b.pcap
│   │   ├── gmailchat1.pcap
│   │   ├── gmailchat2.pcap
│   │   └── gmailchat3.pcap
│   ├── streaming
│   │   ├── vimeo1.pcap
│   │   ├── vimeo2.pcap
│   │   ├── vimeo3.pcap
│   │   ├── vimeo4.pcap
│   │   ├── youtube1.pcap
│   │   ├── youtube2.pcap
│   │   ├── youtube3.pcap
│   │   ├── youtube4.pcap
│   │   ├── youtube5.pcap
│   │   ├── youtube6.pcap
│   │   └── youtubeHTML5_1.pcap
│   ├── voip
│   │   ├── voipbuster1b.pcap
│   │   ├── voipbuster2b.pcap
│   │   ├── voipbuster3b.pcap
│   │   ├── voipbuster_4a.pcap
│   │   └── voipbuster_4b.pcap
│   ├── vpn-audio
│   │   ├── vpn_facebook_audio2.pcap
│   │   ├── vpn_hangouts_audio1.pcap
│   │   ├── vpn_hangouts_audio2.pcap
│   │   ├── vpn_skype_audio1.pcap
│   │   └── vpn_skype_audio2.pcap
│   ├── vpn-chat
│   │   ├── vpn_aim_chat1a.pcap
│   │   ├── vpn_aim_chat1b.pcap
│   │   ├── vpn_facebook_chat1a.pcap
│   │   ├── vpn_facebook_chat1b.pcap
│   │   ├── vpn_hangouts_chat1a.pcap
│   │   ├── vpn_hangouts_chat1b.pcap
│   │   ├── vpn_icq_chat1a.pcap
│   │   ├── vpn_icq_chat1b.pcap
│   │   ├── vpn_skype_chat1a.pcap
│   │   └── vpn_skype_chat1b.pcap
│   ├── vpn-file
│   │   ├── vpn_ftps_A.pcap
│   │   ├── vpn_ftps_B.pcap
│   │   ├── vpn_sftp_A.pcap
│   │   ├── vpn_sftp_B.pcap
│   │   ├── vpn_skype_files1a.pcap
│   │   └── vpn_skype_files1b.pcap
│   ├── vpn-mail
│   │   ├── vpn_email2a.pcap
│   │   └── vpn_email2b.pcap
│   ├── vpn-streaming
│   │   ├── vpn_vimeo_A.pcap
│   │   ├── vpn_vimeo_B.pcap
│   │   └── vpn_youtube_A.pcap
│   └── vpn-voip
│       ├── vpn_voipbuster1a.pcap
│       └── vpn_voipbuster1b.pcap
├── X-APP
│   ├── aimchat
│   │   ├── AIMchat1.pcap
│   │   ├── AIMchat2.pcap
│   │   ├── aim_chat_3a.pcap
│   │   ├── aim_chat_3b.pcap
│   │   ├── extra_AIMchat5.pcap
│   │   ├── vpn_aim_chat1a.pcap
│   │   └── vpn_aim_chat1b.pcap
│   ├── amazon
│   │   ├── amazon.pcap
│   │   ├── yamaxun__browse.pcap
│   │   ├── yamaxun__search.pcap
│   │   └── yamaxun__start.pcap
│   ├── baidu
│   │   ├── baidu__overall.pcap
│   │   ├── extra_baidu__browse.pcap
│   │   ├── extra_baidu__search.pcap
│   │   ├── extra_baidu__start.pcap
│   │   └── extra_baidu__vedio.pcap
│   ├── cloudmusic
│   │   ├── cloudmusic__overall.pcap
│   │   ├── extra_cloudmusic__listen.pcap
│   │   └── extra_cloudmusic__search.pcap
│   ├── facebook
│   │   └── video2b.pcap
│   ├── gmail
│   │   ├── gmailchat1.pcap
│   │   ├── gmailchat2.pcap
│   │   └── gmailchat3.pcap
│   ├── google
│   │   └── torGoogle.pcap
│   ├── itunes
│   │   ├── iTunes__browse.pcap
│   │   ├── iTunes__download.pcap
│   │   ├── iTunes.pcap
│   │   └── iTunes__search.pcap
│   ├── jd
│   │   ├── extra_jingdong__search.pcap
│   │   ├── extra_jingdong__start.pcap
│   │   ├── jingdong__browse.pcap
│   │   └── jingdong__overall.pcap
│   ├── kugou
│   │   ├── extra_KGService__free.pcap
│   │   ├── extra_KGService__listen.pcap
│   │   ├── extra_KGService__MV.pcap
│   │   └── KGService__overall.pcap
│   ├── MS-Exchange
│   │   ├── MS-Exchange__login.pcap
│   │   ├── MS-Exchange__overall.pcap
│   │   ├── MS-Exchange__readmail.pcap
│   │   └── MS-Exchange__sendmail.pcap
│   ├── mssql
│   │   ├── extra_mssql_connect.pcap
│   │   └── extra_mssql_execsql.pcap
│   ├── netflix
│   │   ├── extra_netflix2.pcap
│   │   └── netflix1.pcap
│   ├── pplive
│   │   ├── extra_PPAP__search.pcap
│   │   ├── extra_PPAP__watch.pcap
│   │   ├── extra_PPLive__live.pcap
│   │   ├── PPAP__download.pcap
│   │   ├── PPAP__live.pcap
│   │   ├── PPAP__overall.pcap
│   │   ├── PPLive__overall.pcap
│   │   └── PPLive__search.pcap
│   ├── qq
│   │   ├── extra_QQ__login.pcap
│   │   ├── extra_QQ__qqspace.pcap
│   │   ├── extra_QQ__videocall.pcap
│   │   ├── extra_QQ__voicecall.pcap
│   │   ├── QQ__overall.pcap
│   │   └── QQ__sendandreceive.pcap
│   ├── sinauc
│   │   └── SinaUC__overall.pcap
│   ├── skype
│   │   ├── skype_audio3.pcap
│   │   ├── Skype__overall.pcap
│   │   └── skype_video1b.pcap
│   ├── sohu
│   │   ├── extra_SHRes__overall.pcap
│   │   ├── extra_SHRes__search.pcap
│   │   ├── extra_SHRes__start.pcap
│   │   ├── extra_SoHuVA__watch.pcap
│   │   ├── SoHuVA__download.pcap
│   │   └── SoHuVA__overall.pcap
│   ├── spotify
│   │   ├── extra_spotify2.pcap
│   │   ├── extra_spotify3.pcap
│   │   └── spotify1.pcap
│   ├── taobao
│   │   ├── extra_taobao__browse.pcap
│   │   ├── extra_taobao__search.pcap
│   │   └── taobao.pcap
│   ├── thunder
│   │   └── ThunderPlatform__bt.pcap
│   ├── tudou
│   │   ├── TudouVa__download.pcap
│   │   └── TudouVa__overall.pcap
│   ├── twitter
│   │   └── torTwitter.pcap
│   ├── vimeo
│   │   ├── extra_vimeo2.pcap
│   │   ├── extra_vimeo3.pcap
│   │   └── vimeo4.pcap
│   ├── voipbuster
│   │   └── extra_vpn_voipbuster1a.pcap
│   ├── weibo
│   │   ├── extra_xinlangweibo__browse.pcap
│   │   ├── extra_xinlangweibo__search.pcap
│   │   ├── extra_xinlangweibo__start.pcap
│   │   ├── extra_xinlangweibo__vedio.pcap
│   │   └── xinlangweibo__overall.pcap
│   ├── yahoomail
│   │   ├── YAHOOM~1__overall.pcap
│   │   ├── yahoomail__download.pcap
│   │   ├── yahoomail__login.pcap
│   │   ├── yahoomail__overall.pcap
│   │   ├── yahoomail__readmail.pcap
│   │   └── yahoomail__sendmail.pcap
│   ├── youku
│   │   ├── extra_ikuacc__download.pcap
│   │   ├── ikuacc__overall.pcap
│   │   ├── ikuacc__watch.pcap
│   │   ├── YoukuDesktop__overall.pcap
│   │   └── youkupage__overall.pcap
│   └── youtube
│       ├── extra_youtube1.pcap
│       ├── extra_youtube2.pcap
│       ├── extra_youtube3.pcap
│       ├── extra_youtube6.pcap
│       └── youtube4.pcap
├── X-WEB
│   ├── amazon
│   │   └── amazon_1.pcap
│   ├── baidu
│   │   ├── baidu_1.pcap
│   │   └── baidu_2.pcap
│   ├── bing
│   │   ├── bing_1.pcap
│   │   └── bing_2.pcap
│   ├── douban
│   │   └── douban_1.pcap
│   ├── facebook
│   │   ├── facebook_1.pcap
│   │   └── facebook_2.pcap
│   ├── google
│   │   ├── google_1.pcap
│   │   └── google_2.pcap
│   ├── imdb
│   │   ├── imdb_1.pcap
│   │   └── imdb_2.pcap
│   ├── instagram
│   │   ├── instagram_1.pcap
│   │   └── instagram_2.pcap
│   ├── iqiyi
│   │   └── iqiyi_1.pcap
│   ├── jd
│   │   └── JD_1.pcap
│   ├── neteasemusic
│   │   ├── NeteaseMusic_1.pcap
│   │   └── NeteaseMusic_2.pcap
│   ├── qqmail
│   │   ├── qqmail_1.pcap
│   │   └── qqmail_2.pcap
│   ├── reddit
│   │   ├── reddit_1.pcap
│   │   └── reddit_2.pcap
│   ├── taobao
│   │   └── taobao_1.pcap
│   ├── ted
│   │   ├── TED_1.pcap
│   │   └── TED_2.pcap
│   ├── tieba
│   │   └── tieba_1.pcap
│   ├── twitter
│   │   └── twitter_1.pcap
│   ├── weibo
│   │   └── weibo_1.pcap
│   ├── youku
│   │   └── youku_1.pcap
│   └── youtube
│       └── youtube_1.pcap
└── USTC-TFC
    ├── Cridex
    │   └── Cridex.pcap
    ├── Geodo
    │   └── Geodo.pcap
    ├── Htbot
    │   └── Htbot.pcap
    ├── Miuref
    │   └── Miuref.pcap
    ├── Neris
    │   └── Neris.pcap
    ├── Nsis-ay
    │   └── Nsis-ay.pcap
    ├── Shifu
    │   └── Shifu.pcap
    ├── Tinba
    │   └── Tinba.pcap
    ├── Virut
    │   └── Virut.pcap
    └── Zeus
        └── Zeus.pcap
```
