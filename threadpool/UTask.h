#ifndef CGRAPH_GCLUSTER_H
#define CGRAPH_GCLUSTER_H

#include "CObject.h"
#include <iostream>
#include <unistd.h>

class UTask: public CObject {
public:
    explicit UTask(int i) : index_(i) {}
    int run() override {
        std::cout << "UTask running: " << index_ << std::endl;
        sleep(1);
        std::cout << "UTask finished: " << index_ << std::endl;
        return index_;
    }
private:
    int index_ {0};
};

#endif //CGRAPH_GCLUSTER_H
