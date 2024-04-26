#include "lock.h"
#include "threadpool.h"

int main() {
    {
        runLock();
    }
    {
        runThreadPool();
    }
}