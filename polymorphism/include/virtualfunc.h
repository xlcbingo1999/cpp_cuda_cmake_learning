
namespace Polymorphism {
    class Shape {
    public:
        static Shape* StaticCreate();
        virtual Shape* Create(); // 协变函数
        virtual void area(); // 纯虚函数
        virtual void height();   // 虚函数
    };

    class Rectangle: public Shape {
    public:
        static Rectangle* StaticCreate();
        // 这里的override不能省略, final表示此时这个函数无法被Override
        Rectangle* Create() override final;
        void area() override; // 重写OverRide
        void height() override;
    };

    class Circle: public Shape {
    public:
        static Circle* StaticCreate();
        Circle* Create() override final;
        void area() override;
        void height() override;
    };
};