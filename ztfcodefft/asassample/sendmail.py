# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:19:10 2022

@author: dingxu
"""

import yagmail
 
print('hello world')

# 登录你的邮箱
yag = yagmail.SMTP(user = 'dx409@11.com', password = 'dingxu409', host = 'smtp.qq.com')

# 发送邮件
yag.send(to = ['dx409@qq.ac.cn'], subject = '主题', contents = ['内容', 'it is ok!'])
