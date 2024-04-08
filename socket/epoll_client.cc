#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define IP_ADDR "127.0.0.1"
#define IP_PORT 8888

int main() {
    int socket_fd = socket(AF_INET, SOCK_STREAM, 0); 
    if (socket_fd == -1) {
        perror("socket error");
        exit(1);
    }

    int opt = 1;
    setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));

    addr.sin_family = AF_INET;
    addr.sin_port = htons(IP_PORT);
    addr.sin_addr.s_addr = inet_addr(IP_ADDR);
    int ret = connect(socket_fd, (struct sockaddr *) &addr, sizeof(addr));
    if (ret == -1) {
        perror("connect error");
        exit(1);
    }

    char buf[1024]{};
    strcpy(buf, "Hello World");
    int len = send(socket_fd, buf, strlen(buf), 0);
    if (len < 0) {
        perror("send error");
        exit(1);
    }

    len = recv(socket_fd, buf, sizeof(buf), 0);
    if (len < 0) {
        perror("recv error");
        exit(1);
    }

    printf("client recv: %s\n", buf);

    close(socket_fd);
    return 0;
}