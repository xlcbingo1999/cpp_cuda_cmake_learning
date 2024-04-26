void func(int a, int b);
int runThreadPool();

class MyTemp {
private:
    float m_c = 2.0f;
public:
    // 这种写法会出现不可预期的竞态条件
    void mp(float d);
};