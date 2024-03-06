# 一个Cmake的测试工程

> 参考“单身剑法传人”的教程

## 准备

- Linux环境下需要安装部分依赖包 
  - [doxygen](https://www.doxygen.nl/)
  - [mono](https://www.mono-project.com/)


## Configure and Build

```shell
cmake -S . -B -DMONO_PATH=<mono的下载位置>

cmake --build build
```