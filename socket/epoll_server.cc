#include <netinet/in.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <errno.h>
#include <string>


#define IP_ADDR "127.0.0.1"
#define IP_PORT 8888
#define EPOLL_COUNT 100
#define SOCKET_COUNT 64

#define EPOLL_LT 0
#define EPOLL_ET 1
#define FD_BLOCK 0
#define FD_NONBLOCK 1
#define MAX_BUFFER_SIZE 8

int set_nonblock(int fd) {
    printf("fd %d: set_nonblock\n", fd);
    int old_flags = fcntl(fd, F_GETFL);
    fcntl(fd, F_SETFL, old_flags | O_NONBLOCK);
    return old_flags;
}

void addfd_to_epoll(int epoll_fd, int fd, int epoll_type, int block_type) {
    struct epoll_event ep_event;
    ep_event.data.fd = fd;

    // 表示事件为EPOLLIN, EPOLLIN事件则只有当对端有数据写入时才会触发
    // 所以触发一次后需要不断读取所有数据直到读完EAGAIN为止
    // 则剩下的数据只有在下次对端有写入时才能一起取出来了。
    ep_event.events = EPOLLIN;

    // 边缘触发, 设置event的标识符    
    if (epoll_type == EPOLL_ET) {
        ep_event.events |= EPOLLET;
    } 

    // 阻塞读写
    if (block_type == FD_NONBLOCK) {
        set_nonblock(fd);
    }

    // 开启控制
    // 表示事件注册函数, 这里注册的是ADD操作, epoll_wait返回的条件一定是epoll_ctl添加的感兴趣的op
    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &ep_event);
}

void error_exit(const char *msg) {
    perror(msg);
    exit(1);
}

void epoll_lt(int sock_fd) {
    // 水平触发, 如果从文件缓冲区里面读不到的时候就直接关闭了
    char buffer[MAX_BUFFER_SIZE]{};

    memset(buffer, 0, MAX_BUFFER_SIZE);
    printf("begin epoll_lt(%d)\n", sock_fd);

    int ret = recv(sock_fd, buffer, MAX_BUFFER_SIZE, 0);
    if (ret > 0) {
        printf("recv: %s, all %d bytes\n", buffer, ret);
        // 这里再加一个写的逻辑即可
    } else if (ret == 0) {
        printf("client close sock_fd: %d\n", sock_fd);
        close(sock_fd);
    } else {
        error_exit("recv err");
    }

    printf("end epoll_lt(%d)\n", sock_fd);
}

void epoll_et_loop(int sock_fd) {
    // 可以一次性完成某个输入的读取, 但此时无法退出这个函数
    // 直到所有的数据读完才能退出这个函数, 一般来说是客户端关闭了
    char buffer[MAX_BUFFER_SIZE]{};


    printf("begin epoll_et_loop(%d)\n", sock_fd);
    while (true) {
        memset(buffer, 0, MAX_BUFFER_SIZE);
        int ret = recv(sock_fd, buffer, MAX_BUFFER_SIZE, 0); // 这里就是I/O操作
        if (ret == -1) {
            // 只有读到了-1, 才表示此时已经完成了读取
            // 注意: 非阻塞IO的场景中, 如果没有数据可以读的时候, 会马上返回这里
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                printf("recv all data\n");
            } else {
                close(sock_fd);
            }
            break;
        } else if (ret == 0) {
            printf("client close sock_fd: %d\n", sock_fd);
            close(sock_fd);
            break;
        } else {
            printf("recv: %s, all %d bytes\n", buffer, ret);
        }
    }
    printf("end epoll_et_loop(%d)\n", sock_fd);
}

void epoll_et_noloop(int sock_fd) {
    // 目前只能读取到数据的其中一部分, 然后程序会马上返回
    // 直到下一次从客户端发出的事件到达的时候, 才会接着进行读取
    char buffer[MAX_BUFFER_SIZE]{};


    printf("begin epoll_et_noloop(%d)\n", sock_fd);

    memset(buffer, 0, MAX_BUFFER_SIZE);
    int ret = recv(sock_fd, buffer, MAX_BUFFER_SIZE, 0);
    if (ret > 0) {
        printf("recv: %s, all %d bytes\n", buffer, ret);
    } else if (ret == 0) {
        printf("client close sock_fd: %d\n", sock_fd);
        close(sock_fd);
    }  else {
        close(sock_fd);
    }

    printf("end epoll_et_noloop(%d)\n", sock_fd);
}

void epoll_process(int epoll_fd, struct epoll_event *events, int num, int sock_fd, 
                int epoll_type, int block_type, int et_with_loop) {
    struct sockaddr_in client_addr;
    
    
    for (int i = 0; i < num; i++) {
        int new_fd = events[i].data.fd; // 轮询

        if (new_fd == sock_fd) {
            // 此时第一次有一个连接来和服务端进行建立
            printf("====== fisrt round accept() ======\n");
            printf("begin accept()...\n");

            // 用于模拟一个繁忙的服务器
            printf("accept() sleep 3...\n");
            sleep(3);
            socklen_t client_addrlen = sizeof(client_addr);
            int conn_fd = accept(sock_fd, (struct sockaddr *) &client_addr, &client_addrlen);
            printf("get conn_fd: %d\n", conn_fd);

            // 两阶段socket, 需要将accept后的socket注册到epoll中
            addfd_to_epoll(epoll_fd, conn_fd, epoll_type, block_type);
            printf("end accept()... \n");
            
        } else if (events[i].events & EPOLLIN) {
            // 此时服务端读取到了相关的数据
            
            if (epoll_type == EPOLL_LT) {
                printf("===== begin EPOLL_LT =====\n");
                epoll_lt(new_fd);
            } else if (epoll_type == EPOLL_ET) {
                printf("===== begin EPOLL_ET =====\n");
                if (et_with_loop) {
                    epoll_et_loop(new_fd);
                } else {
                    epoll_et_noloop(new_fd);
                }
            }
        } else {
            printf("[warning] other\n");
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        error_exit("argc error");
    }

    int global_epoll_type = 0;
    int global_et_with_loop = 0;
    if (std::string(argv[1]) == "lt") {
        global_epoll_type = 0;
    } else if (std::string(argv[1]) == "et") {
        global_epoll_type = 1;
        
        if (std::string(argv[3]) == "loop") {
            global_et_with_loop = 1;
        } else if (std::string(argv[3]) == "nloop") {
            global_et_with_loop = 0;
        } else {
            error_exit("global_et_with_loop error");
        }
    } else {
        error_exit("global_epoll_type error");
    }

    int global_fd_block_type = 0;
    if (std::string(argv[2]) == "block") {
        global_fd_block_type = 0;
    } else if (std::string(argv[2]) == "nblock") {
        global_fd_block_type = 1;
    } else {
        error_exit("global_fd_block_type error");
    }

    // 创建socket文件描述符
    int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd == -1) {
        error_exit("socket error");
    }

    // 端口复用
    int opt = 1;
    setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET; // Address Family 地址族
    addr.sin_port = htons(IP_PORT); // 设置端口
    addr.sin_addr.s_addr = htonl(INADDR_ANY); // 监听any域名

    // 讲socket文件描述符和ip:port进行bind
    int ret = bind(sock_fd, (struct sockaddr *) &addr, sizeof(addr));
    if (ret == -1) {
        perror("bind error");
        exit(1);
    }

    // socket文件描述符需要开始listen
    ret = listen(sock_fd, SOCKET_COUNT);
    if (ret == -1) {
        perror("listem error");
        exit(1);
    }

    // 创建epoll
    int epoll_fd = epoll_create(EPOLL_COUNT);
    if (epoll_fd == -1) {
        perror("epoll_create error");
        exit(1);
    }


    // 水平触发阻塞
    addfd_to_epoll(epoll_fd, sock_fd, global_epoll_type, global_fd_block_type); 
    // 水平触发非阻塞
    // addfd_to_epoll(epoll_fd, sock_fd, EPOLL_LT, FD_NONBLOCK); 

    // 边缘触发阻塞
    // addfd_to_epoll(epoll_fd, sock_fd, EPOLL_ET, FD_BLOCK);
    // 边缘触发非阻塞
    // addfd_to_epoll(epoll_fd, sock_fd, EPOLL_ET, FD_NONBLOCK);

    struct epoll_event events[EPOLL_COUNT];
    int size = sizeof(events) / sizeof(struct epoll_event);

    while (true) {
        printf("epoll_wait ... \n");

        // 阻塞等待注册事件的发生
        int num = epoll_wait(epoll_fd, events, size, -1);
        
        
        // 默认是 水平触发LT + 文件阻塞读写
        epoll_process(epoll_fd, events, num, sock_fd, global_epoll_type, global_fd_block_type, global_et_with_loop);

        // 水平触发LT + 文件非阻塞读写
        // epoll_process(epoll_fd, events, num, sock_fd, EPOLL_LT, FD_NONBLOCK);

        // 边缘触发ET + 文件阻塞读写
        

        // 可以修改成 边缘触发(Edge Triangle): 只会从epoll_wait苏醒一次, 必须一次性把内核缓冲区数据读完(一般是非阻塞I/O, 程序循环读写直到错误发生)
    }

    close(epoll_fd);
    close(sock_fd);

    return 0;
}

// 监听sock_fd: 使用水平触发, 避免有些客户端连接不上
// 读写conn_fd: 
//    水平触发: 阻塞和非阻塞没有什么区别, 建议非阻塞
//    边缘触发: 一定要用非阻塞I/O, 且一定需要用while循环一次性把所有的数据都读完