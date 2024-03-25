#include "virtualfunc.h"
#include <iostream>

namespace Polymorphism
{
    Shape* Shape::StaticCreate() {
        std::cout << "Shape::Create" << std::endl;
        return new Shape();
    }

    Shape* Shape::Create() {
        std::cout << "Shape::Create" << std::endl;
        return new Shape();
    }

    void Shape::area() {
        std::cout << "Shape::area" << std::endl;
    }

    void Shape::height() {
        std::cout << "Shape::height" << std::endl;
    }

    Rectangle* Rectangle::StaticCreate() {
        std::cout << "Rectangle::Create" << std::endl;
        return new Rectangle();
    }

    Rectangle* Rectangle::Create() {
        std::cout << "Rectangle::Create" << std::endl;
        return new Rectangle();
    }

    void Rectangle::height() {
        std::cout << "Rectangle::height" << std::endl;
    }

    void Rectangle::area() {
        std::cout << "Rectangle::area" << std::endl;
    }

    Circle* Circle::StaticCreate() {
        std::cout << "Circle::Create" << std::endl;
        return new Circle();
    }

    Circle* Circle::Create() {
        std::cout << "Circle::Create" << std::endl;
        return new Circle();
    }

    void Circle::height() {
        std::cout << "Circle::height" << std::endl;
    }

    void Circle::area() {
        std::cout << "Circle::area" << std::endl;
    }
};
