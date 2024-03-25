#include "pod.h"
#include <cstring>
#include <iostream>

namespace RAII {
    PodClass* GetClass(int _a, int _b, char _initC) {
        PodClass* pod = new PodClass();
        pod->a = _a;
        pod->b = _b;
        memset(pod->addr, _initC, 50 * sizeof(char));

        {
            char newstr[10];
            newstr[9] = '\0';
            memcpy(newstr, pod->addr+10, 9);
            std::cout << "newstr: " << newstr << std::endl;
        }

        return pod;
    }
};