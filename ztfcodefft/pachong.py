# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 21:33:20 2021

@author: dingxu
"""

import requests
import re
# 下载一个网页
url = 'http://tessebs.villanova.edu/search_results?order_by=period&then_order_by=none&incat_only=on&bjd0_min=&bjd0_max=&teff_min=&teff_max=&period_min=&period_max=&logg_min=&logg_max=&morph_min=&morph_max=&abun_min=&abun_max=&ra_min=&ra_max=&tmag_min=&tmag_max=&dec_min=&dec_max=&sec_min=1&sec_max=&glon_min=&glon_max=&nsec_min=&nsec_max=&glat_min=&glat_max=&sigid_min=&sigid_max=&display_format=html'
# 模拟浏览器发送http请求
response = requests.get(url)
# 编码方式
response.encoding='utf-8'
# 目标小说主页的网页源码
html = response.text
print(html)
