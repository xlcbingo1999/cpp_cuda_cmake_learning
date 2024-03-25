
namespace RAII {
    // 都是public继承, 所有的成员函数和成员变量都是public的
    // 不能使用模板参数
    struct PodClass {
        int a;
        float b;
        char addr[50];
    };

    PodClass* GetClass(int _a, int _b, char _initC);
    
};