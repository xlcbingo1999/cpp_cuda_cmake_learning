#include <iostream>

// 奇异递归模板模式（ Curiously Recurring Template Pattern，CRTP）是一种 C++ 模板编程技术，
// 用于实现静态多态。它通过在模板类中使用自身作为模板参数来实现，从而允许在编译时确定类型信息。

// 这是一种很好的静态多态的实现方式，目的是可以少写点代码

template <typename DerivedVector>
struct VectorBase {
    DerivedVector& underlying() {
        return static_cast<DerivedVector&>(*this);
    }


    const DerivedVector& underlying() const {
        return static_cast<const DerivedVector&>(*this);
    }

    inline DerivedVector& operator+=(const DerivedVector& rhs) {
        this->underlying() += rhs; // 这个操作会调用派生类的 operator+= 函数
        return this->underlying();
    }
};

struct Vector3: VectorBase<Vector3> {
    float x, y, z;
    Vector3() = default;

    Vector3(float x, float y, float z) : x(x), y(y), z(z) {}
    inline Vector3& operator+=(const Vector3& rhs) {
        this->x += rhs.x;
        this->y += rhs.y;
        this->z += rhs.z;
        return *this;
    }
};

// 这种方式可以实现该函数在vector2， vector3， vector4，... 都能适配
template <typename T>
inline void SelfAddOp(VectorBase<T>& lhs, const VectorBase<T>& rhs) {
    lhs += rhs.underlying();
}

int main() {
    Vector3 v0(1, 2, 3);
    Vector3 v1(4, 5, 6);
    Vector3 v2(7, 8, 9);
    SelfAddOp(v0, v1);
    SelfAddOp(v0, v2);

    std::cout << v0.x << " " << v0.y << " " << v0.z << std::endl;
}