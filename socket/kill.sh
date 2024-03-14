###################################################################
# File Name: kill.sh
# Author: whang1234
# mail: @163.com
# Created Time: 2023年12月04日 星期一 14时01分58秒
#=============================================================
#!/bin/bash
sudo lsof -t -i:9000 | xargs -r sudo kill -9
sudo lsof -t -i:9001 | xargs -r sudo kill -9
sudo lsof -t -i:9002 | xargs -r sudo kill -9

