/*
 * @FilePath: \SimSwap\docs\js\which-image.js
 * @Author: Ziang Liu
 * @Date: 2021-07-03 16:34:56
 * @LastEditors: Ziang Liu
 * @LastEditTime: 2021-07-03 16:44:10
 * Copyright (C) 2021 SJTU. All rights reserved.
 */



function select_source(number) {
    var items = ['anni', 'chenglong', 'zhoujielun', 'zhuyin'];
    var item_id = items[number];

    for (i = 0; i < 4; i++) {
        if (number == i) {
            document.getElementById(items[i]).style.borderWidth = '5px';
            document.getElementById(items[i]).style.borderColor = 'red';
            document.getElementById(items[i]).style.borderStyle = 'outset';
        } else {
            document.getElementById(items[i]).style.border = 'none';
        }
    }
    document.getElementById('jiroujinlun').src = './img/' + item_id + '.webp';

}