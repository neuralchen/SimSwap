#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: logo_class.py
# Created Date: Tuesday June 29th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 11th October 2021 12:39:55 am
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################

class logo_class:
    
    @staticmethod
    def print_group_logo():
        logo_str = """

███╗   ██╗██████╗ ███████╗██╗ ██████╗     ███████╗     ██╗████████╗██╗   ██╗
████╗  ██║██╔══██╗██╔════╝██║██╔════╝     ██╔════╝     ██║╚══██╔══╝██║   ██║
██╔██╗ ██║██████╔╝███████╗██║██║  ███╗    ███████╗     ██║   ██║   ██║   ██║
██║╚██╗██║██╔══██╗╚════██║██║██║   ██║    ╚════██║██   ██║   ██║   ██║   ██║
██║ ╚████║██║  ██║███████║██║╚██████╔╝    ███████║╚█████╔╝   ██║   ╚██████╔╝
╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝     ╚══════╝ ╚════╝    ╚═╝    ╚═════╝ 
Neural Rendering Special Interesting Group of SJTU
                                                                            
        """
        print(logo_str)

    @staticmethod
    def print_start_training():
        logo_str = """
   _____  __                __     ______              _         _              
  / ___/ / /_ ____ _ _____ / /_   /_  __/_____ ____ _ (_)____   (_)____   ____ _
  \__ \ / __// __ `// ___// __/    / /  / ___// __ `// // __ \ / // __ \ / __ `/
 ___/ // /_ / /_/ // /   / /_     / /  / /   / /_/ // // / / // // / / // /_/ / 
/____/ \__/ \__,_//_/    \__/    /_/  /_/    \__,_//_//_/ /_//_//_/ /_/ \__, /  
                                                                       /____/   
        """
        print(logo_str)

if __name__=="__main__":
    # logo_class.print_group_logo()
    logo_class.print_start_training()